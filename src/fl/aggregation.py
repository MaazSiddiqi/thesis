"""Federated aggregation strategies.

Extracted from reference FL_ResNet_HAM10000.py: FedAvg().
See docs/phase-01/migration-map.md.
"""

import copy
from typing import Dict, List

import torch


StateDict = Dict[str, torch.Tensor]


def fed_avg(state_dicts: List[StateDict]) -> StateDict:
    """Compute the unweighted average of a list of model state dicts (FedAvg)."""
    if not state_dicts:
        raise ValueError("fed_avg requires at least one state dict")
    w_avg = copy.deepcopy(state_dicts[0])
    for key in w_avg.keys():
        for i in range(1, len(state_dicts)):
            w_avg[key] += state_dicts[i][key]
        w_avg[key] = torch.div(w_avg[key], len(state_dicts))
    return w_avg
