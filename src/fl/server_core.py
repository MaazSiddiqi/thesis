"""FL server-side orchestration logic (transport-agnostic).

See docs/phase-01/architecture.md (FL Server) and migration-map.md.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.fl.aggregation import fed_avg, StateDict

log = logging.getLogger(__name__)


class FLServer:
    """Coordinates federated rounds: holds global model and aggregates updates."""

    def __init__(self, model: nn.Module, num_clients: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_clients = num_clients
        self.current_round = 0
        self._pending_updates: List[StateDict] = []

    def get_global_state_dict(self) -> StateDict:
        return self.model.state_dict()

    def submit_update(self, client_id: int, state_dict: StateDict) -> Optional[StateDict]:
        """Accept a client update. Returns the new global state dict when all
        expected clients for this round have reported, otherwise None."""
        log.info("Round %d: received update from client %d (%d/%d)",
                 self.current_round, client_id,
                 len(self._pending_updates) + 1, self.num_clients)
        self._pending_updates.append(state_dict)

        if len(self._pending_updates) >= self.num_clients:
            return self._aggregate()
        return None

    def _aggregate(self) -> StateDict:
        log.info("Round %d: aggregating %d updates", self.current_round, len(self._pending_updates))
        new_state = fed_avg(self._pending_updates)
        self.model.load_state_dict(new_state)
        self._pending_updates = []
        self.current_round += 1
        return new_state
