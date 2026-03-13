"""Transport-agnostic interfaces for FL server and client communication.

See docs/phase-01/architecture.md (Transport abstraction) and decision-log.md ADR-2.
Implementations live in adapter modules (e.g. http_adapter.py).
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch

StateDict = Dict[str, torch.Tensor]


class ServerTransport(ABC):
    """Interface that an FL server exposes to accept client traffic."""

    @abstractmethod
    def start(self) -> None:
        """Start listening for client connections (blocking or background)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Gracefully shut down the transport."""
        ...


class ClientTransport(ABC):
    """Interface a client uses to talk to the FL server."""

    @abstractmethod
    def get_global_model(self) -> StateDict:
        """Pull the current global model state dict from the server."""
        ...

    @abstractmethod
    def submit_update(self, client_id: int, state_dict: StateDict) -> StateDict:
        """Push a local update and receive the new global model after aggregation.

        The server may block until all clients have reported for the current
        round, then return the aggregated model.
        """
        ...
