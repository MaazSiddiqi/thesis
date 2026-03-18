"""Per-round network and training statistics collection for FL clients.

RoundStats holds counters/timings for one FL round.
StatsCollector accumulates them and prints a summary table at the end.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import time


@dataclass
class RoundStats:
    round_num: int
    get_model_bytes: int = 0
    get_model_ms: float = 0.0
    submit_bytes: int = 0
    submit_ms: float = 0.0
    train_ms: float = 0.0
    poll_count: int = 0
    wait_ms: float = 0.0
    wire_tx_bytes: int = 0
    wire_rx_bytes: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    eval_loss: float = 0.0
    eval_acc: float = 0.0


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

    def record_wire_bytes(self, tx: int, rx: int) -> None:
        if self._current is not None:
            self._current.wire_tx_bytes = tx
            self._current.wire_rx_bytes = rx

    def record_train_metrics(self, loss: float, acc: float) -> None:
        if self._current is not None:
            self._current.train_loss = loss
            self._current.train_acc = acc

    def record_eval_metrics(self, loss: float, acc: float) -> None:
        if self._current is not None:
            self._current.eval_loss = loss
            self._current.eval_acc = acc

    def record_round_wall(self, wall_ms: float) -> None:
        """Compute and store wait_ms once the full round wall-clock time is known.

        Called from the transport layer after submit_update() receives the new
        global model.  At that point get_model_ms, train_ms, and submit_ms are
        all already recorded, so wait_ms can be derived immediately.
        """
        if self._current is not None:
            r = self._current
            r.wait_ms = max(
                0.0,
                wall_ms - r.get_model_ms - r.train_ms - r.submit_ms,
            )

    def finish_round(self) -> None:
        if self._current is not None:
            self._rounds.append(self._current)
            self._current = None

    def all_stats(self) -> List[RoundStats]:
        return list(self._rounds)

    def log_summary(self, client_id: int, logger) -> None:
        """Log a formatted summary table to the given logger."""
        sep = "─" * 114
        logger.info("── Network Stats Summary " + "─" * 90)
        header = (
            f"{'round':>6}  {'get_model_ms':>12}  {'submit_ms':>9}  "
            f"{'train_ms':>8}  {'wait_ms':>7}  {'submit_bytes':>12}  {'poll_count':>10}  "
            f"{'wire_tx_B':>10}  {'wire_rx_B':>10}"
        )
        logger.info(header)
        for rs in self._rounds:
            row = (
                f"{rs.round_num:>6}  {rs.get_model_ms:>12.1f}  {rs.submit_ms:>9.1f}  "
                f"{rs.train_ms:>8.1f}  {rs.wait_ms:>7.1f}  {rs.submit_bytes:>12d}  {rs.poll_count:>10d}  "
                f"{rs.wire_tx_bytes:>10d}  {rs.wire_rx_bytes:>10d}"
            )
            logger.info(row)
        logger.info(sep)

    def save_json(
        self,
        filepath: str | Path,
        client_id: int,
        experiment_name: str,
        config: Dict[str, Any],
    ) -> None:
        """Write all collected stats to a JSON file for post-processing."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": experiment_name,
            "client_id": client_id,
            "config": config,
            "rounds": [asdict(rs) for rs in self._rounds],
        }
        filepath.write_text(json.dumps(payload, indent=2))
