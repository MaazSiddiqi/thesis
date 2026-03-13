"""Unit tests for state_dict serialization roundtrip. See docs/phase-01/test-plan.md."""

import torch

from src.transport.serialization import state_dict_to_bytes, bytes_to_state_dict


def test_roundtrip_simple():
    sd = {"w": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([0.5])}
    data = state_dict_to_bytes(sd)
    assert isinstance(data, bytes)
    assert len(data) > 0
    restored = bytes_to_state_dict(data)
    assert torch.allclose(sd["w"], restored["w"])
    assert torch.allclose(sd["b"], restored["b"])


def test_roundtrip_resnet():
    from src.models.resnet import build_resnet18
    model = build_resnet18(num_classes=7)
    sd = model.state_dict()
    data = state_dict_to_bytes(sd)
    restored = bytes_to_state_dict(data)
    for key in sd:
        assert torch.allclose(sd[key], restored[key]), f"Mismatch at key {key}"
