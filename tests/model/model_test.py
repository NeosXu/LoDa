import os

import torch

from src.model import Model, create_model
from src.utils.loss import get_loss
from tests.test_case import ProjectTestCase


class TestModel(ProjectTestCase):
    @classmethod
    def setup_class(cls):
        cls.model_input = torch.rand(4, 3, 224, 224)
        cls.model_target = torch.randn(4) * 4 + 1

    def setup_method(self, method):
        super(TestModel, self).setup_method()
        self.net = create_model(cfg=self.cfg)
        self.loss_f = get_loss(cfg=self.cfg)
        self.model = Model(self.cfg, self.net, self.loss_f)

    def test_model(self):
        assert self.model.cfg == self.cfg
        assert self.model.net == self.net
        assert self.model.loss_f == self.loss_f

    def test_run_network(self):
        output = self.model.run_network(self.model_input.to(self.cfg.dist.device))
        assert output.shape == self.model_target.unsqueeze(1).shape

    def test_optimize_parameters(self):
        self.model.optimize_parameters(
            self.model_input.to(self.cfg.dist.device),
            self.model_target.to(self.cfg.dist.device),
        )
        assert self.model.log.loss_v is not None

    def test_inference(self):
        output = self.model.inference(self.model_input.to(self.cfg.dist.device))
        assert output.shape == self.model_target.unsqueeze(1).shape

    def test_save_load_network(self):
        local_net = create_model(cfg=self.cfg)
        self.loss_f = get_loss(cfg=self.cfg)
        local_model = Model(self.cfg, local_net, self.loss_f)

        self.model.save_network()
        save_filename = "%s_%d.pt" % (self.cfg.name, self.model.step)
        save_path = os.path.join(self.cfg.log.chkpt_dir, save_filename)
        self.cfg.load.network_chkpt_path = save_path

        assert os.path.exists(save_path) and os.path.isfile(save_path)

        local_model.load_network()
        parameters = zip(
            list(local_model.net.parameters()), list(self.model.net.parameters())
        )
        for load, origin in parameters:
            assert (load == origin).all()

    def test_save_load_state(self):
        local_net = create_model(cfg=self.cfg)
        self.loss_f = get_loss(cfg=self.cfg)
        local_model = Model(self.cfg, local_net, self.loss_f)

        self.model.save_training_state()
        save_filename = "%s_%d.state" % (self.cfg.name, self.model.step)
        save_path = os.path.join(self.cfg.log.chkpt_dir, save_filename)
        self.cfg.load.resume_state_path = save_path

        assert os.path.exists(save_path) and os.path.isfile(save_path)

        local_model.load_training_state()
        parameters = zip(
            list(local_model.net.parameters()), list(self.model.net.parameters())
        )
        for load, origin in parameters:
            assert (load == origin).all()
        assert local_model.epoch == self.model.epoch
        assert local_model.step == self.model.step
