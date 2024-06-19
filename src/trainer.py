import datetime
import itertools
import os
import random
import traceback

import hydra
import pyrootutils
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataset import DataloaderMode, create_dataloader
from src.model import Model, create_model
from src.tools import test_model, train_model
from src.utils.loss import get_loss
from src.utils.utils import get_logger, is_logging_process, set_random_seed
from src.utils.writer import Writer


def setup(cfg, rank):
    os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.dist.master_port
    timeout_sec = 1800
    if cfg.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    # initialize the process group
    dist.init_process_group(
        cfg.dist.mode,
        rank=rank,
        world_size=cfg.dist.gpus,
        timeout=timeout,
    )


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, cfg):
    mp.spawn(fn, args=(cfg,), nprocs=cfg.dist.gpus, join=True)


def train_loop(rank, cfg):
    logger = get_logger(cfg, os.path.basename(__file__))
    if cfg.dist.device == "cuda" and cfg.dist.gpus != 0:
        cfg.dist.device = rank
        setup(cfg, rank)
        torch.cuda.set_device(cfg.dist.device)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    # setup writer
    if is_logging_process():
        # set log/checkpoint dir
        os.makedirs(cfg.log.chkpt_dir, exist_ok=True)
        # set writer (tensorboard / wandb)
        writer = Writer(cfg, "tensorboard")
        cfg_str = OmegaConf.to_yaml(cfg)
        logger.info("Config:\n" + cfg_str)
        if cfg.data.root == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")
    else:
        writer = None

    # make dataloader
    if is_logging_process():
        logger.info("Making train dataloader...")
    train_loader, train_sampler = create_dataloader(cfg, DataloaderMode.train, rank)
    if is_logging_process():
        logger.info("Making test dataloader...")
    test_loader, _ = create_dataloader(cfg, DataloaderMode.test, rank)

    # init Model
    net_arch = create_model(cfg=cfg)
    loss_f = get_loss(cfg=cfg)
    model = Model(cfg, net_arch, loss_f, rank)

    # load training state / network checkpoint
    if cfg.load.resume_state_path is not None:
        model.load_training_state()
    elif cfg.load.network_chkpt_path is not None:
        model.load_network()
    else:
        if is_logging_process():
            logger.info("Starting new training run.")

    try:
        if (
            cfg.dist.device == "cpu"
            or cfg.dist.gpus == 0
            or cfg.data.divide_dataset_per_gpu
        ):
            epoch_step = 1
        else:
            epoch_step = cfg.dist.gpus
        for model.epoch in itertools.count(model.epoch + 1, epoch_step):
            if model.epoch >= cfg.num_epoch:
                break
            if train_sampler is not None:
                train_sampler.set_epoch(model.epoch)
            train_model(cfg, model, train_loader, writer)
            if model.epoch % cfg.log.net_chkpt_interval == 0:
                model.save_network()
            if model.epoch % cfg.log.train_chkpt_interval == 0:
                model.save_training_state()
            test_model(cfg, model, test_loader, writer)
        if is_logging_process():
            logger.info("End of Train")
    except Exception:
        if is_logging_process():
            logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.device == "cuda" and cfg.dist.gpus != 0:
            cleanup()


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(hydra_cfg: DictConfig):
    hydra_cfg.dist.device = hydra_cfg.dist.device.lower()
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
        hydra_cfg.hydra_output_dir = HydraConfig.get().run.dir
    # random seed
    if hydra_cfg.random_seed is None:
        hydra_cfg.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.random_seed)

    if hydra_cfg.dist.device == "cuda" and hydra_cfg.dist.gpus < 0:
        hydra_cfg.dist.gpus = torch.cuda.device_count()
    if hydra_cfg.dist.device == "cpu" or hydra_cfg.dist.gpus == 0:
        hydra_cfg.dist.gpus = 0
        train_loop(0, hydra_cfg)
    else:
        distributed_run(train_loop, hydra_cfg)


if __name__ == "__main__":
    main()
