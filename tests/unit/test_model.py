"""Unit tests for ResNet18 model. See docs/phase-01/test-plan.md."""

import torch

from src.models.resnet import build_resnet18


def test_forward_shape():
    model = build_resnet18(num_classes=7)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 7)


def test_forward_shape_different_classes():
    model = build_resnet18(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 10)


def test_state_dict_keys_nonempty():
    model = build_resnet18(num_classes=7)
    sd = model.state_dict()
    assert len(sd) > 0
    assert any("conv1" in k for k in sd.keys())
