"""Unit tests for IID data partitioning. See docs/phase-01/test-plan.md."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.data.partitioning import dataset_iid


def _make_dataset(n=100):
    return TensorDataset(torch.randn(n, 1), torch.zeros(n))


def test_partition_sizes():
    ds = _make_dataset(100)
    parts = dataset_iid(ds, num_users=5)
    assert len(parts) == 5
    for idxs in parts.values():
        assert len(idxs) == 20


def test_partitions_are_disjoint():
    ds = _make_dataset(100)
    parts = dataset_iid(ds, num_users=4)
    all_idxs = set()
    for idxs in parts.values():
        assert all_idxs.isdisjoint(idxs)
        all_idxs.update(idxs)


def test_partitions_cover_dataset():
    ds = _make_dataset(80)
    parts = dataset_iid(ds, num_users=4)
    all_idxs = set()
    for idxs in parts.values():
        all_idxs.update(idxs)
    assert all_idxs == set(range(80))
