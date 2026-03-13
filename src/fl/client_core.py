"""FL client-side training logic (transport-agnostic).

Extracted and refactored from reference FL_ResNet_HAM10000.py: LocalUpdate.
See docs/phase-01/architecture.md (FL Client) and migration-map.md.
"""

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.metrics import calculate_accuracy


StateDict = Dict[str, torch.Tensor]


class LocalTrainer:
    """Encapsulates one FL client's local training and evaluation."""

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 1e-4,
        local_epochs: int = 1,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = torch.device(device)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, model: nn.Module) -> Tuple[StateDict, float, float]:
        """Train *model* on local data and return (state_dict, avg_loss, avg_acc)."""
        model = copy.deepcopy(model).to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        epoch_losses, epoch_accs = [], []
        for _ in range(self.local_epochs):
            batch_losses, batch_accs = [], []
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                fx = model(images)
                loss = self.loss_fn(fx, labels)
                acc = calculate_accuracy(fx, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                batch_accs.append(acc.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
            epoch_accs.append(sum(batch_accs) / len(batch_accs))

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_acc = sum(epoch_accs) / len(epoch_accs)
        return model.state_dict(), avg_loss, avg_acc

    def evaluate(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate *model* on local test data; return (avg_loss, avg_acc)."""
        model = copy.deepcopy(model).to(self.device)
        model.eval()
        batch_losses, batch_accs = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                fx = model(images)
                loss = self.loss_fn(fx, labels)
                acc = calculate_accuracy(fx, labels)
                batch_losses.append(loss.item())
                batch_accs.append(acc.item())
        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_acc = sum(batch_accs) / len(batch_accs)
        return avg_loss, avg_acc
