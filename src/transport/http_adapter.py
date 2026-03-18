"""HTTP+JSON transport adapter for Federated Learning.

Implements the transport interfaces from interface.py using FastAPI (server)
and httpx (client). Model state dicts are serialized as binary blobs via
torch.save and transferred as application/octet-stream.

Round synchronization uses a polling pattern: clients POST their update,
then poll GET /round_status until the server's round number advances past
the round the client submitted for. The round number is the single source
of truth — it increments inside FLServer._aggregate() when FedAvg runs,
so no separate boolean flag is needed.

See docs/decision-log.md ADR-1 and architecture.md (HTTP+JSON Adapter).
"""

import logging
import threading
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

from src.fl.server_core import FLServer
from src.stats.collector import StatsCollector
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
                "pending": len(self.fl_server._pending_updates),
                "expected": self.fl_server.num_clients,
            }

        @self.app.post("/update/{client_id}")
        async def post_update(client_id: int, request: Request):
            body = await request.body()
            sd = bytes_to_state_dict(body)
            # Capture the round number BEFORE submit_update, because _aggregate()
            # increments current_round. The client uses this to detect completion.
            submitted_round = self.fl_server.current_round
            self.fl_server.submit_update(client_id, sd)
            return {"status": "accepted", "submitted_round": submitted_round}

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

    def __init__(
        self,
        server_url: str,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
        stats: Optional[StatsCollector] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._stats = stats
        self._round_start: Optional[float] = None

    def _fetch_global_model(self) -> StateDict:
        """Raw model fetch without stats recording (used internally by submit_update)."""
        resp = httpx.get(f"{self.server_url}/model", timeout=self.timeout)
        resp.raise_for_status()
        return bytes_to_state_dict(resp.content)

    def get_global_model(self) -> StateDict:
        # _round_start anchors the full round wall-clock so that wait_ms can be
        # computed once submit_update() returns the new global model.
        self._round_start = time.monotonic()
        resp = httpx.get(f"{self.server_url}/model", timeout=self.timeout)
        resp.raise_for_status()
        if self._stats is not None:
            elapsed_ms = (time.monotonic() - self._round_start) * 1000
            self._stats.record_get_model(len(resp.content), elapsed_ms)
        return bytes_to_state_dict(resp.content)

    def submit_update(self, client_id: int, state_dict: StateDict) -> StateDict:
        data = state_dict_to_bytes(state_dict)
        submit_start = time.monotonic()
        # No timeout on the POST body: a bandwidth-limited client may take many
        # minutes to upload a large state dict, and we must not cut it short.
        resp = httpx.post(
            f"{self.server_url}/update/{client_id}",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=None,
        )
        resp.raise_for_status()
        submit_elapsed_ms = (time.monotonic() - submit_start) * 1000
        submitted_round = resp.json()["submitted_round"]

        # Poll until the server's round number advances past the round we submitted
        # for. The round number is the single source of truth: it increments inside
        # FLServer._aggregate() when FedAvg runs, with no separate flag to manage.
        # If the round does not complete within self.timeout, raise TimeoutError —
        # this is intentional: a very slow client causing others to wait IS a result.
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            status = httpx.get(f"{self.server_url}/round_status", timeout=10).json()
            if self._stats is not None:
                self._stats.record_poll()
            if status["round"] > submitted_round:
                result = self._fetch_global_model()
                if self._stats is not None:
                    self._stats.record_submit(len(data), submit_elapsed_ms)
                    if self._round_start is not None:
                        round_wall_ms = (time.monotonic() - self._round_start) * 1000
                        self._stats.record_round_wall(round_wall_ms)
                return result
            time.sleep(self.poll_interval)

        raise TimeoutError("Timed out waiting for round to complete")
