import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.utils import get_logger, is_logging_process


class Model:
    def __init__(self, cfg, net_arch, loss_f, rank=0):
        # prepare var
        self.cfg = cfg
        self.device = self.cfg.dist.device
        self.net = net_arch.to(self.device)
        self.rank = rank
        if self.device != "cpu" and self.cfg.dist.gpus != 0:
            self.net = DDP(self.net, device_ids=[self.rank])
        self.step = 0
        self.epoch = -1
        self._logger = get_logger(cfg, os.path.basename(__file__))

        # init optimizer
        optimizer_mode = self.cfg.optimizer.name
        if optimizer_mode == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), **(self.cfg.optimizer.param)
            )
        elif optimizer_mode == "adamW":
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), **(self.cfg.optimizer.param)
            )
        else:
            raise Exception("%s optimizer not supported" % optimizer_mode)

        # init scheduler
        scheduler_mode = self.cfg.scheduler.name
        if scheduler_mode == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **(self.cfg.scheduler.param)
            )
        else:
            raise Exception("%s scheduler not supported" % scheduler_mode)

        # init loss
        self.loss_f = loss_f
        self.log = OmegaConf.create()
        self.log.loss_v = 0

    def optimize_parameters(self, model_input, model_target):
        self.net.train()
        self.optimizer.zero_grad()
        output = self.run_network(model_input)
        loss_v = self.compute_loss(output, model_target)
        loss_v.backward()
        self.optimizer.step()
        self.scheduler.step()
        # set log
        self.log.loss_v = loss_v.detach().item()

    def compute_loss(self, model_output, model_target):
        loss_v = 0.0
        for fn, weight in self.loss_f:
            loss_v += weight * fn(
                model_output, model_target.unsqueeze(1).to(self.device)
            )
        return loss_v

    def inference(self, model_input):
        self.net.eval()
        output = self.run_network(model_input)
        return output

    def run_network(self, model_input):
        model_input = model_input.to(self.device)
        output = self.net(model_input)
        return output

    def save_network(self, save_file=True):
        if is_logging_process():
            net = self.net.module if isinstance(self.net, DDP) else self.net
            state_dict = net.obtain_state_to_save()
            for module_name, module_param in state_dict.items():
                if isinstance(module_param, torch.Tensor):
                    state_dict[module_name] = module_param.to("cpu")
                else:
                    for key, param in module_param.items():
                        state_dict[module_name][key] = param.to("cpu")
            if save_file:
                save_filename = "%s_%d.pt" % (self.cfg.name, self.step)
                save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
                torch.save(state_dict, save_path)
                if self.cfg.log.use_wandb and self.cfg.log.wandb_save_model:
                    wandb.save(save_path)
                if is_logging_process():
                    self._logger.info("Saved network checkpoint to: %s" % save_path)
            return state_dict

    def load_network(self, loaded_net=None):
        add_log = False
        if loaded_net is None:
            add_log = True
            if self.cfg.load.wandb_load_path is not None:
                self.cfg.load.network_chkpt_path = wandb.restore(
                    self.cfg.load.network_chkpt_path,
                    run_path=self.cfg.load.wandb_load_path,
                ).name
            loaded_net = torch.load(
                self.cfg.load.network_chkpt_path,
                map_location=torch.device(self.device),
            )
        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v

        if isinstance(self.net, DDP):
            self.net.module.load_saved_state(
                loaded_clean_net, strict=self.cfg.load.strict_load
            )
        else:
            self.net.load_saved_state(
                loaded_clean_net, strict=self.cfg.load.strict_load
            )
        if is_logging_process() and add_log:
            self._logger.info(
                "Checkpoint %s is loaded" % self.cfg.load.network_chkpt_path
            )

    def save_training_state(self):
        if is_logging_process():
            save_filename = "%s_%d.state" % (self.cfg.name, self.step)
            save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
            net_state_dict = self.save_network(False)
            state = {
                "model": net_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            }
            torch.save(state, save_path)
            if self.cfg.log.use_wandb and self.cfg.log.wandb_save_model:
                wandb.save(save_path)
            if is_logging_process():
                self._logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self):
        if self.cfg.load.wandb_load_path is not None:
            self.cfg.load.resume_state_path = wandb.restore(
                self.cfg.load.resume_state_path,
                run_path=self.cfg.load.wandb_load_path,
            ).name
        resume_state = torch.load(
            self.cfg.load.resume_state_path,
            map_location=torch.device(self.device),
        )

        self.load_network(loaded_net=resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.scheduler.load_state_dict(resume_state["scheduler"])
        self.step = resume_state["step"]
        self.epoch = resume_state["epoch"]
        if is_logging_process():
            self._logger.info(
                "Resuming from training state: %s" % self.cfg.load.resume_state_path
            )
