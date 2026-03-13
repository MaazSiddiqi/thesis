# Network-Aware Federated Learning Simulator

Docker-based FL simulator that runs an arbitrary number of clients under
configurable network conditions (bandwidth, latency, congestion) and reports
network statistics at the end of training.

**Thesis**: *Network-Aware Analysis of Federated Learning*
(CS4490Z, Western University).

## Repository Layout

```
src/                  # Modular Python package
  models/             #   ResNet18 architecture
  data/               #   Dataset, partitioning, preprocessing
  training/           #   Metrics helpers
  fl/                 #   FL core: aggregation, client, server (transport-agnostic)
  transport/          #   Transport interfaces + HTTP+JSON adapter
  server_main.py      #   FL server entrypoint
  client_main.py      #   FL client entrypoint
infra/                # Docker infrastructure
  Dockerfile.ml-base  #   Python + PyTorch + network tools image
  Dockerfile.network-test  # Lightweight network-only image
  compose.yml         #   FL server + clients + iperf3
  scripts/            #   Connectivity and tc/netem validation scripts
tests/                # pytest unit and integration tests
docs/                 # Architecture, decisions, test plan, runbook
old/                  # Archived reference scripts (original SplitFed codebase)
```

## Quick Start

See [docs/runbook.md](docs/runbook.md) for full instructions.

```bash
# Run unit + integration tests (no Docker needed)
python3 -m pytest tests/ -v

# Build and start containers
docker compose -f infra/compose.yml build
docker compose -f infra/compose.yml up

# Network validation (tc/netem/iperf3)
docker compose -f infra/compose.yml --profile network-test up -d
./infra/scripts/validate_network.sh
```

## Roadmap

- **Phase 1** (done): Container foundation, FL core, HTTP transport, network utility validation.
- **Phase 2** (next): Arbitrary N-client orchestration, experiment configuration, network statistics collection.
- **Phase 3** (planned): Full experiment matrix (bandwidth/latency/congestion sweeps) and analysis.

## Reference

The original SplitFed codebase (Thapa et al. 2022) is archived under `old/`.
The original thesis proposal covering FL, SL, and SplitFed is at
[thesis_proposal.tex](thesis_proposal.tex) — retained for academic context.
