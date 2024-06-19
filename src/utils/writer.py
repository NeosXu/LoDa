import os

import wandb
from torch.utils.tensorboard import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, cfg, logdir):
        self.cfg = cfg
        if cfg.log.use_tensorboard:
            self.tensorboard = SummaryWriter(logdir)
        if cfg.log.use_wandb:
            wandb_init_conf = cfg.log.wandb_init_conf
            wandb.init(config=cfg, **wandb_init_conf)
            if cfg.log.wandb_init_conf.save_code:
                wandb.run.log_code("./src/")
            wandb.save(os.path.join(cfg.hydra_output_dir, ".hydra/*.yaml"))

    def logging_with_step(self, value, step, logging_name):
        if self.cfg.log.use_tensorboard:
            self.tensorboard.add_scalar(logging_name, value, step)
        if self.cfg.log.use_wandb:
            wandb.log({logging_name: value}, step=step)
