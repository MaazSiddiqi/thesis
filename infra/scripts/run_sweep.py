#!/usr/bin/env python3
"""Run the full QoS experiment sweep for the thesis.

Generates per-experiment YAML configs, runs each via Docker Compose,
and saves logs + JSON stats to results/<experiment_name>/.

Idempotent: each successful experiment writes a .done checkpoint
containing a SHA-256 hash of its config. On re-run, experiments whose
checkpoint hash matches the current config are skipped automatically.
Use --force to re-run everything regardless.

Usage:
    python infra/scripts/run_sweep.py              # run (resumes from last checkpoint)
    python infra/scripts/run_sweep.py --dry-run     # print plan without running
    python infra/scripts/run_sweep.py --only bw     # run only bandwidth sweep
    python infra/scripts/run_sweep.py --skip bw-1   # skip the 1 Mbps run
    python infra/scripts/run_sweep.py --force        # ignore checkpoints, re-run all
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATE_SCRIPT = REPO_ROOT / "infra" / "scripts" / "generate_compose.py"
COMPOSE_FILE = REPO_ROOT / "infra" / "compose.generated.yml"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "_logs"

SWEEP_DEFAULTS = {
    "rounds": 20,
    "num_clients": 3,
    "alpha": 0.5,
    "port": 8000,
    "bandwidth": "100mbit",
    "latency": "5ms",
    "loss": "0%",
}


def make_config(
    name: str,
    rounds: int,
    num_clients: int,
    alpha: float,
    port: int,
    clients: List[Dict[str, Any]],
) -> dict:
    return {
        "experiment": {
            "name": name,
            "rounds": rounds,
            "num_clients": num_clients,
            "alpha": alpha,
        },
        "server": {"port": port},
        "clients": clients,
    }


def uniform_clients(
    n: int,
    bandwidth: str,
    latency: str,
    loss: str,
) -> List[Dict[str, Any]]:
    """All clients share the same network conditions."""
    return [
        {
            "id": i + 1,
            "network": {
                "bandwidth": bandwidth,
                "latency": latency,
                "loss": loss,
            },
        }
        for i in range(n)
    ]


def build_sweep_matrix() -> List[Dict[str, Any]]:
    """Return the ordered list of experiments to run.

    One-variable-at-a-time design:
      - Bandwidth sweep: vary BW, hold latency=5ms loss=0%
      - Latency sweep:   vary latency, hold BW=100mbit loss=0%
      - Loss sweep:      vary loss, hold BW=100mbit latency=5ms
      - Straggler:       2 fast + 1 slow client
      - Scale:           5 and 10 clients at baseline
    """
    d = SWEEP_DEFAULTS
    experiments = []

    # --- Bandwidth sweep ---
    for bw_val in [1, 5, 10, 50, 100]:
        bw_str = f"{bw_val}mbit"
        name = f"bw-{bw_val}"
        rounds = 5 if bw_val == 1 else d["rounds"]
        experiments.append({
            "name": name,
            "group": "bw",
            "config": make_config(
                name=name,
                rounds=rounds,
                num_clients=d["num_clients"],
                alpha=d["alpha"],
                port=d["port"],
                clients=uniform_clients(d["num_clients"], bw_str, d["latency"], d["loss"]),
            ),
            "note": f"BW={bw_str}, {rounds}r" + (" (reduced rounds — ~1h estimated)" if bw_val == 1 else ""),
        })

    # --- Latency sweep (skip 5ms — already in bw-100) ---
    for lat_val in [50, 200]:
        lat_str = f"{lat_val}ms"
        name = f"lat-{lat_val}"
        experiments.append({
            "name": name,
            "group": "lat",
            "config": make_config(
                name=name,
                rounds=d["rounds"],
                num_clients=d["num_clients"],
                alpha=d["alpha"],
                port=d["port"],
                clients=uniform_clients(d["num_clients"], d["bandwidth"], lat_str, d["loss"]),
            ),
            "note": f"latency={lat_str}",
        })

    # --- Loss sweep (skip 0% — already in bw-100) ---
    for loss_val in [1, 5]:
        loss_str = f"{loss_val}%"
        name = f"loss-{loss_val}"
        experiments.append({
            "name": name,
            "group": "loss",
            "config": make_config(
                name=name,
                rounds=d["rounds"],
                num_clients=d["num_clients"],
                alpha=d["alpha"],
                port=d["port"],
                clients=uniform_clients(d["num_clients"], d["bandwidth"], d["latency"], loss_str),
            ),
            "note": f"loss={loss_str}",
        })

    # --- Straggler experiment ---
    experiments.append({
        "name": "straggler",
        "group": "straggler",
        "config": make_config(
            name="straggler",
            rounds=d["rounds"],
            num_clients=3,
            alpha=d["alpha"],
            port=d["port"],
            clients=[
                {"id": 1, "network": {"bandwidth": "100mbit", "latency": "5ms", "loss": "0%"}},
                {"id": 2, "network": {"bandwidth": "100mbit", "latency": "5ms", "loss": "0%"}},
                {"id": 3, "network": {"bandwidth": "5mbit", "latency": "50ms", "loss": "1%"}},
            ],
        ),
        "note": "2 fast + 1 slow (5mbit/50ms/1%)",
    })

    # --- Scale experiments ---
    for n in [5, 10]:
        name = f"scale-{n}"
        experiments.append({
            "name": name,
            "group": "scale",
            "config": make_config(
                name=name,
                rounds=d["rounds"],
                num_clients=n,
                alpha=d["alpha"],
                port=d["port"],
                clients=uniform_clients(n, d["bandwidth"], d["latency"], d["loss"]),
            ),
            "note": f"{n} clients at baseline",
        })

    return experiments


def config_hash(config: dict) -> str:
    """Deterministic SHA-256 of the experiment config dict."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def checkpoint_path(name: str) -> Path:
    return RESULTS_DIR / name / ".done"


def is_done(exp: Dict[str, Any]) -> bool:
    """Return True if this experiment already ran with the same config."""
    cp = checkpoint_path(exp["name"])
    if not cp.exists():
        return False
    stored_hash = cp.read_text().strip().split("\n")[0]
    return stored_hash == config_hash(exp["config"])


def write_checkpoint(exp: Dict[str, Any], elapsed_secs: float) -> None:
    cp = checkpoint_path(exp["name"])
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(
        f"{config_hash(exp['config'])}\n"
        f"completed: {datetime.now().isoformat()}\n"
        f"elapsed: {timedelta(seconds=int(elapsed_secs))}\n"
    )


def print_plan(experiments: List[Dict[str, Any]], force: bool = False) -> None:
    print(f"\n{'#':>3}  {'Name':<16} {'Group':<10} {'Status':<10} {'Note'}")
    print("─" * 80)
    for i, exp in enumerate(experiments, 1):
        done = is_done(exp) and not force
        status = "SKIP" if done else "PENDING"
        print(f"{i:>3}  {exp['name']:<16} {exp['group']:<10} {status:<10} {exp['note']}")
    pending = sum(1 for e in experiments if not is_done(e) or force)
    skipped = len(experiments) - pending
    print(f"\nTotal: {len(experiments)} experiments ({pending} to run, {skipped} already done)\n")


def run_one(exp: Dict[str, Any], idx: int, total: int) -> bool:
    """Run a single experiment. Returns True on success."""
    name = exp["name"]
    config = exp["config"]

    print(f"\n{'='*60}")
    print(f"  [{idx}/{total}] Running: {name}")
    print(f"  {exp['note']}")
    print(f"{'='*60}\n")

    # Clean up any stale containers from a previous crash
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
        capture_output=True,
    )

    config_path = REPO_ROOT / "experiments" / f"_sweep-{name}.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    try:
        subprocess.run(
            [sys.executable, str(GENERATE_SCRIPT), str(config_path)],
            check=True,
        )

        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "build", "--quiet"],
            check=True,
        )

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / f"{name}.log"
        start = time.monotonic()

        with open(log_file, "w") as lf:
            result = subprocess.run(
                ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "--abort-on-container-exit"],
                stdout=lf,
                stderr=subprocess.STDOUT,
            )

        elapsed = time.monotonic() - start
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # Cleanup containers
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
            capture_output=True,
        )

        if result.returncode == 0:
            write_checkpoint(exp, elapsed)
            print(f"  DONE  {name}  ({elapsed_str})  ✓ checkpointed")
            return True
        else:
            print(f"  FAIL  {name}  (exit code {result.returncode}, {elapsed_str})")
            print(f"        Log: {log_file}")
            return False

    except Exception as exc:
        print(f"  ERROR  {name}: {exc}")
        return False
    finally:
        config_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis experiment sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--only", type=str, help="Run only experiments in this group (bw, lat, loss, straggler, scale)")
    parser.add_argument("--skip", type=str, nargs="+", default=[], help="Skip experiments by name (e.g. bw-1)")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoints, re-run all experiments")
    args = parser.parse_args()

    experiments = build_sweep_matrix()

    if args.only:
        experiments = [e for e in experiments if e["group"] == args.only]
    if args.skip:
        experiments = [e for e in experiments if e["name"] not in args.skip]

    print_plan(experiments, force=args.force)

    if args.dry_run:
        print("Dry run — nothing executed.")
        return

    to_run = experiments if args.force else [e for e in experiments if not is_done(e)]

    if not to_run:
        print("All experiments already completed. Use --force to re-run.")
        return

    print(f"Starting sweep: {len(to_run)} experiments to run "
          f"({len(experiments) - len(to_run)} skipped via checkpoint)...")
    print(f"Results → {RESULTS_DIR}/")
    print(f"Logs    → {LOGS_DIR}/\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = []
    sweep_start = time.monotonic()

    for i, exp in enumerate(to_run, 1):
        if run_one(exp, i, len(to_run)):
            successes += 1
        else:
            failures.append(exp["name"])

    total_time = str(timedelta(seconds=int(time.monotonic() - sweep_start)))
    print(f"\n{'='*60}")
    print(f"  Sweep complete: {successes}/{len(to_run)} succeeded  ({total_time})")
    if failures:
        print(f"  Failed: {', '.join(failures)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
