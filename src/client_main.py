"""FL Client entrypoint — runs inside an fl-client-* container.

Usage:
    python -m src.client_main [--server-url http://fl-server:8000]
                              [--client-id 1] [--rounds 1] [--lr 0.0001]

See docs/phase-01/architecture.md (FL Client).
"""

import argparse
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet import build_resnet18
from src.fl.client_core import LocalTrainer
from src.transport.http_adapter import HTTPClientTransport
from src.transport.serialization import bytes_to_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLIENT %(client_id)s] %(message)s")
log = logging.getLogger(__name__)


def _make_synthetic_loaders(num_samples: int = 64, num_classes: int = 7, batch_size: int = 16):
    """Create tiny random data loaders for smoke testing without the real dataset."""
    images = torch.randn(num_samples, 3, 64, 64)
    labels = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(images, labels)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def wait_for_server(url: str, retries: int = 30, delay: float = 2.0):
    """Block until the server's /health endpoint responds."""
    import httpx
    for attempt in range(retries):
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        log.info("Waiting for server (%d/%d)...", attempt + 1, retries)
        time.sleep(delay)
    log.error("Server not reachable after %d attempts", retries)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="FL Client")
    parser.add_argument("--server-url", default=os.environ.get("FL_SERVER_URL", "http://fl-server:8000"))
    parser.add_argument("--client-id", type=int, default=int(os.environ.get("CLIENT_ID", "1")))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("FL_ROUNDS", "1")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("FL_LR", "0.0001")))
    parser.add_argument("--local-epochs", type=int, default=int(os.environ.get("FL_LOCAL_EPOCHS", "1")))
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic data for smoke testing (default for phase-01)")
    args = parser.parse_args()

    # Inject client_id into log format
    old_factory = logging.getLogRecordFactory()
    def record_factory(*a, **kw):
        record = old_factory(*a, **kw)
        record.client_id = args.client_id
        return record
    logging.setLogRecordFactory(record_factory)

    log.info("Client %d starting, server=%s, rounds=%d", args.client_id, args.server_url, args.rounds)

    wait_for_server(args.server_url)

    transport = HTTPClientTransport(server_url=args.server_url)

    train_loader, test_loader = _make_synthetic_loaders()

    trainer = LocalTrainer(
        client_id=args.client_id,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=args.lr,
        local_epochs=args.local_epochs,
        device="cpu",
    )

    model = build_resnet18(num_classes=args.num_classes)

    for rnd in range(args.rounds):
        log.info("Round %d: fetching global model", rnd)
        global_sd = transport.get_global_model()
        model.load_state_dict(global_sd)

        log.info("Round %d: training locally", rnd)
        updated_sd, train_loss, train_acc = trainer.train(model)
        log.info("Round %d: train loss=%.4f acc=%.2f%%", rnd, train_loss, train_acc)

        log.info("Round %d: submitting update to server", rnd)
        new_global_sd = transport.submit_update(args.client_id, updated_sd)
        model.load_state_dict(new_global_sd)

        eval_loss, eval_acc = trainer.evaluate(model)
        log.info("Round %d: eval  loss=%.4f acc=%.2f%%", rnd, eval_loss, eval_acc)

    log.info("Client %d finished %d rounds", args.client_id, args.rounds)


if __name__ == "__main__":
    main()
