#!/usr/bin/env python3
"""Generate infra/compose.generated.yml from an experiment YAML config.

Usage:
    python infra/scripts/generate_compose.py experiments/heterogeneous.yaml
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "infra" / "compose.generated.yml"


def _resolve_config(config_arg: str) -> Path:
    """Resolve the config path: absolute paths pass through, relative paths
    are resolved against the repo root so the script works from any cwd."""
    p = Path(config_arg)
    return p if p.is_absolute() else REPO_ROOT / p


def build_compose(config: dict) -> dict:
    exp = config["experiment"]
    name = exp["name"]
    rounds = exp["rounds"]
    alpha = exp.get("alpha", 0.5)
    port = config["server"]["port"]
    clients = config["clients"]
    num_clients = exp.get("num_clients", len(clients))

    services: dict = {}

    # --- server ---
    services["fl-server"] = {
        "build": {
            "context": "..",
            "dockerfile": "infra/Dockerfile.ml-base",
        },
        "image": "phase1-fl-ml:latest",
        "container_name": "fl-server",
        "hostname": "fl-server",
        "networks": {
            "fl-net": {
                "aliases": ["fl-server"],
            }
        },
        "cap_add": ["NET_ADMIN"],
        "environment": [
            "ROLE=server",
            f"EXPERIMENT_NAME={name}",
            "FL_SERVER_HOST=0.0.0.0",
            f"FL_SERVER_PORT={port}",
            f"NUM_CLIENTS={num_clients}",
            f"FL_ROUNDS={rounds}",
        ],
        "ports": [f"{port}:{port}"],
        "volumes": ["..:/app"],
        "command": ["python", "-m", "src.server_main"],
        "healthcheck": {
            "test": ["CMD", "curl", "-f", f"http://localhost:{port}/health"],
            "interval": "5s",
            "timeout": "3s",
            "retries": 10,
            "start_period": "15s",
        },
    }

    # --- clients ---
    for client in clients:
        cid = client["id"]
        network = client.get("network") or {}
        svc_name = f"fl-client-{cid}"

        env = [
            "ROLE=client",
            f"EXPERIMENT_NAME={name}",
            f"CLIENT_ID={cid}",
            f"FL_SERVER_URL=http://fl-server:{port}",
            f"FL_ROUNDS={rounds}",
            f"NUM_CLIENTS={num_clients}",
            f"FL_ALPHA={alpha}",
        ]
        if network.get("bandwidth") is not None:
            env.append(f"TC_BANDWIDTH={network['bandwidth']}")
        if network.get("latency") is not None:
            env.append(f"TC_LATENCY={network['latency']}")
        if network.get("loss") is not None:
            env.append(f"TC_LOSS={network['loss']}")

        services[svc_name] = {
            "build": {
                "context": "..",
                "dockerfile": "infra/Dockerfile.ml-base",
            },
            "image": "phase1-fl-ml:latest",
            "container_name": svc_name,
            "hostname": svc_name,
            "networks": {
                "fl-net": {
                    "aliases": [svc_name],
                }
            },
            "cap_add": ["NET_ADMIN"],
            "environment": env,
            "volumes": ["..:/app"],
            "command": ["python", "-m", "src.client_main"],
            "depends_on": {
                "fl-server": {"condition": "service_healthy"},
            },
        }

    return {
        "services": services,
        "networks": {
            "fl-net": {
                "driver": "bridge",
                "name": "phase1-fl-net",
            }
        },
    }, name, num_clients


def generate(config_arg: str) -> None:
    config_file = _resolve_config(config_arg)
    with open(config_file) as f:
        config = yaml.safe_load(f)

    compose, name, num_clients = build_compose(config)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

    print(f"Generated infra/compose.generated.yml ({num_clients} clients, experiment: {name})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)
    generate(sys.argv[1])
