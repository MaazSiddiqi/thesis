"""Per-round network and training statistics collection for FL clients.

RoundStats holds counters/timings for one FL round.
StatsCollector accumulates them and prints a summary table at the end.
"""

from dataclasses import dataclass
from typing import List
import time


@dataclass
class RoundStats:
    round_num: int
    get_model_bytes: int = 0       # bytes in GET /model response
    get_model_ms: float = 0.0      # wall-clock ms for GET /model
    submit_bytes: int = 0          # bytes sent in POST /update body
    submit_ms: float = 0.0         # wall-clock ms for POST /update (includes polling)
    train_ms: float = 0.0          # ms spent in local training
    poll_count: int = 0            # number of GET /round_status polls


class StatsCollector:
    def __init__(self):
        self._rounds: List[RoundStats] = []
        self._current: RoundStats | None = None

    def start_round(self, round_num: int) -> None:
        self._current = RoundStats(round_num=round_num)

    def record_get_model(self, byte_count: int, elapsed_ms: float) -> None:
        if self._current is not None:
            self._current.get_model_bytes = byte_count
            self._current.get_model_ms = elapsed_ms

    def record_submit(self, byte_count: int, elapsed_ms: float) -> None:
        if self._current is not None:
            self._current.submit_bytes = byte_count
            self._current.submit_ms = elapsed_ms

    def record_train(self, elapsed_ms: float) -> None:
        if self._current is not None:
            self._current.train_ms = elapsed_ms

    def record_poll(self) -> None:
        if self._current is not None:
            self._current.poll_count += 1

    def finish_round(self) -> None:
        if self._current is not None:
            self._rounds.append(self._current)
            self._current = None

    def all_stats(self) -> List[RoundStats]:
        return list(self._rounds)

    def log_summary(self, client_id: int, logger) -> None:
        """Log a formatted summary table to the given logger."""
        sep = "─" * 72
        logger.info("── Network Stats Summary " + "─" * 48)
        header = (
            f"{'round':>6}  {'get_model_ms':>12}  {'submit_ms':>9}  "
            f"{'train_ms':>8}  {'submit_bytes':>12}  {'poll_count':>10}"
        )
        logger.info(header)
        for rs in self._rounds:
            row = (
                f"{rs.round_num:>6}  {rs.get_model_ms:>12.1f}  {rs.submit_ms:>9.1f}  "
                f"{rs.train_ms:>8.1f}  {rs.submit_bytes:>12d}  {rs.poll_count:>10d}"
            )
            logger.info(row)
        logger.info(sep)
