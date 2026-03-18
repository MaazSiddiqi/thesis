#!/usr/bin/env python3
"""Aggregate experiment JSON results into CSV files for plotting.

Reads results/<experiment>/client-*.json and server.json files produced
by the FL simulator and outputs two CSVs:

  results/client_stats.csv  — one row per (experiment, client, round)
  results/server_stats.csv  — one row per (experiment, round)

Usage:
    python infra/scripts/aggregate_results.py
    python infra/scripts/aggregate_results.py --results-dir results/
"""

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results"

CLIENT_COLUMNS = [
    "experiment", "client_id",
    "bandwidth", "latency", "loss",
    "rounds_total", "num_clients", "alpha",
    "round_num",
    "get_model_ms", "submit_ms", "train_ms", "wait_ms",
    "get_model_bytes", "submit_bytes",
    "poll_count",
    "wire_tx_bytes", "wire_rx_bytes",
    "train_loss", "train_acc",
    "eval_loss", "eval_acc",
]

SERVER_COLUMNS = [
    "experiment", "num_clients", "num_rounds", "round_num", "duration_ms",
]


def collect_client_rows(results_dir: Path) -> list[dict]:
    rows = []
    for json_file in sorted(results_dir.rglob("client-*.json")):
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {json_file}: {e}")
            continue

        exp = data.get("experiment", json_file.parent.name)
        cid = data.get("client_id", 0)
        cfg = data.get("config", {})

        for rd in data.get("rounds", []):
            rows.append({
                "experiment": exp,
                "client_id": cid,
                "bandwidth": cfg.get("bandwidth", ""),
                "latency": cfg.get("latency", ""),
                "loss": cfg.get("loss", ""),
                "rounds_total": cfg.get("rounds", 0),
                "num_clients": cfg.get("num_clients", 0),
                "alpha": cfg.get("alpha", 0.5),
                "round_num": rd.get("round_num", 0),
                "get_model_ms": rd.get("get_model_ms", 0),
                "submit_ms": rd.get("submit_ms", 0),
                "train_ms": rd.get("train_ms", 0),
                "wait_ms": rd.get("wait_ms", 0),
                "get_model_bytes": rd.get("get_model_bytes", 0),
                "submit_bytes": rd.get("submit_bytes", 0),
                "poll_count": rd.get("poll_count", 0),
                "wire_tx_bytes": rd.get("wire_tx_bytes", 0),
                "wire_rx_bytes": rd.get("wire_rx_bytes", 0),
                "train_loss": rd.get("train_loss", 0),
                "train_acc": rd.get("train_acc", 0),
                "eval_loss": rd.get("eval_loss", 0),
                "eval_acc": rd.get("eval_acc", 0),
            })

    return rows


def collect_server_rows(results_dir: Path) -> list[dict]:
    rows = []
    for json_file in sorted(results_dir.rglob("server.json")):
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {json_file}: {e}")
            continue

        exp = data.get("experiment", json_file.parent.name)
        nc = data.get("num_clients", 0)
        nr = data.get("num_rounds", 0)

        for rd in data.get("rounds", []):
            rows.append({
                "experiment": exp,
                "num_clients": nc,
                "num_rounds": nr,
                "round_num": rd.get("round", 0),
                "duration_ms": rd.get("duration_ms", 0),
            })

    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate FL experiment results to CSV")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run experiments first with: python infra/scripts/run_sweep.py")
        return

    print(f"Scanning {results_dir}/ for experiment JSON files...\n")

    client_rows = collect_client_rows(results_dir)
    server_rows = collect_server_rows(results_dir)

    if client_rows:
        write_csv(results_dir / "client_stats.csv", CLIENT_COLUMNS, client_rows)
    else:
        print("  No client JSON files found.")

    if server_rows:
        write_csv(results_dir / "server_stats.csv", SERVER_COLUMNS, server_rows)
    else:
        print("  No server JSON files found.")

    if client_rows:
        experiments = sorted(set(r["experiment"] for r in client_rows))
        print(f"\nExperiments found: {', '.join(experiments)}")


if __name__ == "__main__":
    main()
