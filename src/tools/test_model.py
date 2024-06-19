import os

import numpy as np
import torch
from tqdm import tqdm

from src.utils.metrics import (
    calculate_plcc,
    calculate_rmse,
    calculate_srcc,
    logistic_regression,
)
from src.utils.utils import get_logger, is_logging_process


def test_model(cfg, model, test_loader, writer):
    logger = get_logger(cfg, os.path.basename(__file__))

    model.net.eval()

    if is_logging_process():
        pbar = tqdm(test_loader)
    else:
        pbar = test_loader

    total_test_loss = 0
    pred_scores = []
    gt_scores = []
    test_loop_len = 0
    with torch.no_grad():
        for model_input, model_target in pbar:
            target = model_target.to(cfg.dist.device)

            output = model.inference(model_input)
            loss_v = model.compute_loss(output, target)

            if cfg.dist.gpus > 0:
                # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                torch.distributed.all_reduce(loss_v)
                loss_v /= torch.tensor(float(cfg.dist.gpus))

                # gather scores from all GPUs.
                # TODO: is torch.distributed.all_reduce sequential?
                output_list = [
                    torch.zeros(output.shape, dtype=output.dtype, device=output.device)
                    for _ in range(cfg.dist.device_num)
                ]
                target_list = [
                    torch.zeros(target.shape, dtype=target.dtype, device=target.device)
                    for _ in range(cfg.dist.device_num)
                ]
                torch.distributed.all_gather(output_list, output)
                torch.distributed.all_gather(target_list, target)
                output = torch.concat(output_list)
                target = torch.concat(target_list)

            total_test_loss += loss_v.to("cpu").item()
            pred_scores = pred_scores + output.squeeze(1).cpu().tolist()
            gt_scores = gt_scores + target.cpu().tolist()
            test_loop_len += 1

        total_test_loss /= test_loop_len

        # compute metrics related to Image Quality Assessment task
        pred_scores = np.mean(
            np.reshape(np.array(pred_scores).squeeze(), (-1, cfg.test.patch_num)),
            axis=1,
        )
        gt_scores = np.mean(
            np.reshape(np.array(gt_scores).squeeze(), (-1, cfg.test.patch_num)), axis=1
        )

        test_srcc = calculate_srcc(pred_scores, gt_scores)
        # Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.
        pred_scores = logistic_regression(gt_scores, pred_scores)
        test_plcc = calculate_plcc(pred_scores, gt_scores)
        test_rmse = calculate_rmse(pred_scores, gt_scores)

        if writer is not None:
            writer.logging_with_step(test_plcc, model.step, "test_plcc")
            writer.logging_with_step(test_srcc, model.step, "test_srcc")
            writer.logging_with_step(test_rmse, model.step, "test_rmse")
            writer.logging_with_step(total_test_loss, model.step, "test_loss")
        if is_logging_process():
            logger.info(
                "Test PLCC %.04f at (epoch: %d / step: %d)"
                % (test_plcc, model.epoch + 1, model.step)
            )
            logger.info(
                "Test SRCC %.04f at (epoch: %d / step: %d)"
                % (test_srcc, model.epoch + 1, model.step)
            )
            logger.info(
                "Test RMSE %.04f at (epoch: %d / step: %d)"
                % (test_rmse, model.epoch + 1, model.step)
            )
            logger.info(
                "Test Loss %.04f at (epoch: %d / step: %d)"
                % (total_test_loss, model.epoch + 1, model.step)
            )
