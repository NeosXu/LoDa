import os
import tempfile

import torch
from hydra import compose, initialize

from src.model import create_model

TEST_DIR = tempfile.mkdtemp(prefix="project_tests")


def test_net_arch():
    os.makedirs(TEST_DIR, exist_ok=True)
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default", overrides=[f"working_dir={TEST_DIR}"])

    net = create_model(cfg)

    model_input = torch.rand(4, 3, 224, 224)
    model_target = torch.randn(4) * 4 + 1

    output = net(model_input)
    assert output.shape == model_target.unsqueeze(1).shape
