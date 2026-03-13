#!/usr/bin/env python3
"""Orchestrate a full FL experiment: generate compose, build, then run.

Usage:
    python infra/scripts/run_experiment.py experiments/heterogeneous.yaml

Paths are resolved relative to the repo root so the script works from any cwd.
"""

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATE_SCRIPT = REPO_ROOT / "infra" / "scripts" / "generate_compose.py"
COMPOSE_FILE = REPO_ROOT / "infra" / "compose.generated.yml"


def _resolve_config(config_arg: str) -> Path:
    p = Path(config_arg)
    return p if p.is_absolute() else REPO_ROOT / p


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_path = _resolve_config(sys.argv[1])

    with open(config_path) as f:
        config = yaml.safe_load(f)
    name = config["experiment"]["name"]

    print(f"=== Running experiment: {name} ===")

    # Step 1: Generate compose file
    subprocess.run(
        [sys.executable, str(GENERATE_SCRIPT), str(config_path)],
        check=True,
    )

    # Step 2: Build images
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "build"],
        check=True,
    )

    # Step 3: Run — all containers exit naturally when complete.
    # The server self-terminates after aggregating num_rounds rounds, which
    # then allows docker compose up to return cleanly.
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up"],
        check=True,
    )


if __name__ == "__main__":
    main()
