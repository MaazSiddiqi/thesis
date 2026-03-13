"""HTTP+JSON transport adapter for Federated Learning.

Implements the transport interfaces from interface.py using FastAPI (server)
and httpx (client). Model state dicts are serialized as binary blobs via
torch.save and transferred as application/octet-stream.

Round synchronization uses a polling pattern: clients POST their update,
then poll GET /round_status until the round is complete, then GET /model
for the new global weights. No blocking waits in request handlers.

See docs/phase-01/decision-log.md ADR-1 and architecture.md (HTTP+JSON Adapter).
"""

import logging
import threading
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

from src.fl.server_core import FLServer
from src.transport.interface import ClientTransport, ServerTransport, StateDict
from src.transport.serialization import bytes_to_state_dict, state_dict_to_bytes

log = logging.getLogger(__name__)


class HTTPServerTransport(ServerTransport):
    """FastAPI-based HTTP server that wraps an FLServer instance."""

    def __init__(self, fl_server: FLServer, host: str = "0.0.0.0", port: int = 8000):
        self.fl_server = fl_server
        self.host = host
        self.port = port
        self.app = FastAPI(title="FL Server")
        self._thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._round_complete = False
        self._register_routes()

    def _register_routes(self):
        @self.app.get("/health")
        def health():
            return {"status": "ok", "round": self.fl_server.current_round}

        @self.app.get("/model")
        def get_model():
            sd = self.fl_server.get_global_state_dict()
            data = state_dict_to_bytes(sd)
            return Response(content=data, media_type="application/octet-stream")

        @self.app.get("/round_status")
        def round_status():
            return {
                "round": self.fl_server.current_round,
                "round_complete": self._round_complete,
                "pending": len(self.fl_server._pending_updates),
                "expected": self.fl_server.num_clients,
            }

        @self.app.post("/update/{client_id}")
        async def post_update(client_id: int, request: Request):
            body = await request.body()
            sd = bytes_to_state_dict(body)
            new_state = self.fl_server.submit_update(client_id, sd)
            if new_state is not None:
                self._round_complete = True
            return {"status": "accepted", "round": self.fl_server.current_round}

        @self.app.post("/round_reset")
        def round_reset():
            self._round_complete = False
            return {"status": "reset", "round": self.fl_server.current_round}

    def start(self) -> None:
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", access_log=False,
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
        self._thread.start()
        log.info("HTTP server starting on %s:%d", self.host, self.port)
        time.sleep(1.5)

    def stop(self) -> None:
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)


class HTTPClientTransport(ClientTransport):
    """httpx-based HTTP client that talks to an HTTPServerTransport."""

    def __init__(self, server_url: str, timeout: float = 300.0, poll_interval: float = 0.5):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval

    def get_global_model(self) -> StateDict:
        resp = httpx.get(f"{self.server_url}/model", timeout=self.timeout)
        resp.raise_for_status()
        return bytes_to_state_dict(resp.content)

    def submit_update(self, client_id: int, state_dict: StateDict) -> StateDict:
        data = state_dict_to_bytes(state_dict)
        resp = httpx.post(
            f"{self.server_url}/update/{client_id}",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            status = httpx.get(f"{self.server_url}/round_status", timeout=10).json()
            if status["round_complete"]:
                return self.get_global_model()
            time.sleep(self.poll_interval)

        raise TimeoutError("Timed out waiting for round to complete")

    def notify_round_reset(self):
        resp = httpx.post(f"{self.server_url}/round_reset", timeout=self.timeout)
        resp.raise_for_status()
