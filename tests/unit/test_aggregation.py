"""Unit tests for FedAvg aggregation. See docs/phase-01/test-plan.md."""

import copy

import torch

from src.fl.aggregation import fed_avg


def _make_state_dict(val: float):
    return {"weight": torch.tensor([val, val]), "bias": torch.tensor([val])}


def test_fed_avg_two_dicts():
    sd1 = _make_state_dict(2.0)
    sd2 = _make_state_dict(4.0)
    result = fed_avg([sd1, sd2])
    assert torch.allclose(result["weight"], torch.tensor([3.0, 3.0]))
    assert torch.allclose(result["bias"], torch.tensor([3.0]))


def test_fed_avg_single_dict():
    sd = _make_state_dict(5.0)
    result = fed_avg([sd])
    assert torch.allclose(result["weight"], torch.tensor([5.0, 5.0]))


def test_fed_avg_three_dicts():
    sds = [_make_state_dict(float(i)) for i in [3.0, 6.0, 9.0]]
    result = fed_avg(sds)
    assert torch.allclose(result["weight"], torch.tensor([6.0, 6.0]))


def test_fed_avg_does_not_mutate_inputs():
    sd1 = _make_state_dict(1.0)
    sd2 = _make_state_dict(3.0)
    sd1_orig = copy.deepcopy(sd1)
    fed_avg([sd1, sd2])
    assert torch.allclose(sd1["weight"], sd1_orig["weight"])
