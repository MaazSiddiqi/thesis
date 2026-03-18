"""FL Server entrypoint — runs inside the fl-server container.

Usage:
    python -m src.server_main [--host 0.0.0.0] [--port 8000] [--num-clients 2] [--num-rounds 1]

See docs/architecture.md (FL Server).
"""

import argparse
import logging
import os
import signal
import sys

from src.models.resnet import build_resnet18
from src.fl.server_core import FLServer
from src.transport.http_adapter import HTTPServerTransport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="FL Server")
    parser.add_argument("--host", default=os.environ.get("FL_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FL_SERVER_PORT", "8000")))
    parser.add_argument("--num-clients", type=int, default=int(os.environ.get("NUM_CLIENTS", "2")))
    parser.add_argument("--num-rounds", type=int, default=int(os.environ.get("FL_ROUNDS", "1")))
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    experiment_name = os.environ.get("EXPERIMENT_NAME", "unknown")

    log.info("Building ResNet18 with %d classes", args.num_classes)
    model = build_resnet18(num_classes=args.num_classes)

    fl_server = FLServer(model=model, num_clients=args.num_clients, num_rounds=args.num_rounds)
    transport = HTTPServerTransport(fl_server, host=args.host, port=args.port)

    def shutdown(sig, frame):
        log.info("Shutting down...")
        fl_server.done_event.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info(
        "Experiment=%s Starting FL server on %s:%d (expecting %d clients, %d rounds)",
        experiment_name,
        args.host,
        args.port,
        args.num_clients,
        args.num_rounds,
    )
    transport.start()

    # Block until all rounds are aggregated (or a shutdown signal is received).
    fl_server.done_event.wait()
    # Give any in-flight /round_status polls a brief chance to complete so
    # clients can fetch the final model and log their summaries.
    import time as _time
    _time.sleep(2.0)
    log.info(
        "Experiment=%s Summary (clients=%d, rounds=%d)",
        experiment_name,
        args.num_clients,
        args.num_rounds,
    )
    fl_server.log_summary()

    results_dir = f"/app/results/{experiment_name}"
    fl_server.save_json(f"{results_dir}/server.json", experiment_name)
    log.info("Server stats saved to %s/server.json", results_dir)

    # Give clients time to save their JSON before we exit. The server exiting
    # first triggers docker compose --abort-on-container-exit and SIGTERMs
    # clients before they can persist; delaying here lets clients save first.
    import time as _time
    _time.sleep(15)

    log.info("All %d rounds complete. Shutting down.", args.num_rounds)
    transport.stop()


if __name__ == "__main__":
    main()
