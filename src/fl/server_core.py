"""FL server-side orchestration logic (transport-agnostic).

See docs/architecture.md (FL Server) and migration-map.md.
"""

import logging
import time
from threading import Event
from typing import List, Optional

import torch
import torch.nn as nn

from src.fl.aggregation import fed_avg, StateDict

log = logging.getLogger(__name__)


class RoundRecord:
    """Timing record for one completed aggregation round."""
    def __init__(self, round_num: int, duration_ms: float):
        self.round_num = round_num
        self.duration_ms = duration_ms


class FLServer:
    """Coordinates federated rounds: holds global model and aggregates updates."""

    def __init__(self, model: nn.Module, num_clients: int, num_rounds: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self._pending_updates: List[StateDict] = []
        self._round_start_time: Optional[float] = None
        self._round_records: List[RoundRecord] = []
        # Set when all num_rounds have been aggregated; server_main waits on this.
        self.done_event: Event = Event()

    def get_global_state_dict(self) -> StateDict:
        return self.model.state_dict()

    def submit_update(self, client_id: int, state_dict: StateDict) -> Optional[StateDict]:
        """Accept a client update. Returns the new global state dict when all
        expected clients for this round have reported, otherwise None."""
        if len(self._pending_updates) == 0:
            self._round_start_time = time.monotonic()
        log.info("Round %d: received update from client %d (%d/%d)",
                 self.current_round, client_id,
                 len(self._pending_updates) + 1, self.num_clients)
        self._pending_updates.append(state_dict)

        if len(self._pending_updates) >= self.num_clients:
            return self._aggregate()
        return None

    def _aggregate(self) -> StateDict:
        duration_ms = (time.monotonic() - self._round_start_time) * 1000
        log.info("Round %d: aggregating %d updates (round took %.0f ms)",
                 self.current_round, len(self._pending_updates), duration_ms)
        self._round_records.append(RoundRecord(self.current_round, duration_ms))

        new_state = fed_avg(self._pending_updates)
        self.model.load_state_dict(new_state)
        self._pending_updates = []
        self._round_start_time = None
        self.current_round += 1

        if self.current_round >= self.num_rounds:
            self.done_event.set()

        return new_state

    def log_summary(self) -> None:
        sep = "─" * 44
        log.info("── Server Round Stats Summary " + "─" * 15)
        log.info("%6s  %14s", "round", "duration_ms")
        for r in self._round_records:
            log.info("%6d  %14.1f", r.round_num, r.duration_ms)
        log.info(sep)
