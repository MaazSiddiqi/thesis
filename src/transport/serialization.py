"""Serialize / deserialize PyTorch state dicts for network transfer.

The canonical wire format is a bytes blob produced by torch.save into an
in-memory BytesIO buffer. This avoids lossy JSON encoding of float tensors
and keeps payloads compact.
"""

import io
from typing import Dict

import torch

StateDict = Dict[str, torch.Tensor]


def state_dict_to_bytes(sd: StateDict) -> bytes:
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()


def bytes_to_state_dict(data: bytes) -> StateDict:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)
