"""Integration test: one full FL round on localhost with HTTP transport.

Spins up the HTTP server in-process, runs two clients against it, and checks
that aggregation produces a valid new global model.

Uses a tiny CNN (not ResNet18) to keep serialization/training fast.
See docs/phase-01/test-plan.md (Integration).
"""

import threading
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl.server_core import FLServer
from src.fl.client_core import LocalTrainer
from src.transport.http_adapter import HTTPServerTransport, HTTPClientTransport


NUM_CLIENTS = 2
NUM_SAMPLES = 32
NUM_CLASSES = 7
PORT = 9111


class _TinyModel(nn.Module):
    """Minimal CNN for fast integration tests (~5K params instead of 11M)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _make_loader(n=NUM_SAMPLES):
    images = torch.randn(n, 3, 8, 8)
    labels = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=16)


def _run_client(client_id: int, server_url: str, model_cls, results: dict, errors: list):
    try:
        transport = HTTPClientTransport(server_url=server_url)
        model = model_cls()

        global_sd = transport.get_global_model()
        model.load_state_dict(global_sd)

        trainer = LocalTrainer(
            client_id=client_id,
            train_loader=_make_loader(),
            test_loader=_make_loader(),
            lr=1e-3,
            local_epochs=1,
            device="cpu",
        )
        updated_sd, loss, acc = trainer.train(model)
        new_global = transport.submit_update(client_id, updated_sd)

        results[client_id] = {"loss": loss, "acc": acc, "keys": list(new_global.keys())}
    except Exception as e:
        errors.append((client_id, e))


def test_one_fl_round():
    model = _TinyModel()
    fl_server = FLServer(model=model, num_clients=NUM_CLIENTS, num_rounds=1)
    transport = HTTPServerTransport(fl_server, host="127.0.0.1", port=PORT)

    transport.start()
    time.sleep(1)

    try:
        server_url = f"http://127.0.0.1:{PORT}"
        results: dict = {}
        errors: list = []
        threads = []
        for cid in range(NUM_CLIENTS):
            t = threading.Thread(
                target=_run_client,
                args=(cid, server_url, _TinyModel, results, errors),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Client errors: {errors}"
        assert len(results) == NUM_CLIENTS, f"Expected {NUM_CLIENTS} results, got {len(results)}"
        for cid, res in results.items():
            assert "loss" in res
            assert "acc" in res
            assert len(res["keys"]) > 0

        assert fl_server.current_round == 1
    finally:
        transport.stop()
