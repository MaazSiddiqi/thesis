"""Training metrics helpers.

Extracted from reference FL_ResNet_HAM10000.py: calculate_accuracy().
See docs/phase-01/migration-map.md.
"""

import torch


def calculate_accuracy(fx: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy (percentage) between predictions *fx* and targets *y*."""
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    return 100.0 * correct.float() / preds.shape[0]
