import sys; sys.path.extend(['.', 'src'])
from time import time

import torch

from src.training.motion import FlowSynthesisNetwork
from src.training.layers import generate_coords


def test_apply_optical_flow():
    batch_size = 7
    img_size = 32
    c = 100
    x = torch.randn(batch_size, c, img_size, img_size)
    optical_flow = torch.zeros(batch_size, img_size, img_size, 2)
    y = FlowSynthesisNetwork.apply_optical_flow(x, optical_flow)

    assert torch.allclose(x, y, atol=1e-5)

    optical_flow = torch.ones(batch_size, img_size, img_size, 2) - generate_coords(batch_size, img_size, align_corners=True).permute(0, 2, 3, 1)
    y = FlowSynthesisNetwork.apply_optical_flow(x, optical_flow)
    assert torch.allclose(x[:, :, -1:, -1:].repeat(1, 1, img_size, img_size), y, atol=1e-5)
