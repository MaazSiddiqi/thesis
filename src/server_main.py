"""FL Server entrypoint — runs inside the fl-server container.

Usage:
    python -m src.server_main [--host 0.0.0.0] [--port 8000] [--num-clients 2]

See docs/phase-01/architecture.md (FL Server).
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
    parser.add_argument("--num-classes", type=int, default=7)
    args = parser.parse_args()

    log.info("Building ResNet18 with %d classes", args.num_classes)
    model = build_resnet18(num_classes=args.num_classes)

    fl_server = FLServer(model=model, num_clients=args.num_clients)
    transport = HTTPServerTransport(fl_server, host=args.host, port=args.port)

    def shutdown(sig, frame):
        log.info("Shutting down...")
        transport.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("Starting FL server on %s:%d (expecting %d clients)", args.host, args.port, args.num_clients)
    transport.start()

    # Keep main thread alive
    signal.pause()


if __name__ == "__main__":
    main()
