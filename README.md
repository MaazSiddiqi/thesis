# Network-Aware Federated Learning Simulator

A Docker-based Federated Learning (FL) simulator for studying how network conditions affect distributed training. Clients run in containers with configurable bandwidth, latency, and packet loss via Linux traffic control (`tc netem`). Per-round statistics—upload time, bytes transferred, training time—are collected and exported for analysis.

**Thesis**: *Network-Aware Analysis of Federated Learning* (CS4490Z, Western University).

---

## What This Project Is

This simulator runs real FL training (ResNet-18 on CIFAR-10) over HTTP, with each client optionally shaped to emulate different network profiles. It answers questions like: *How does a 5 Mbit/s client affect round completion time?* and *What is the impact of 1% packet loss on convergence?*

The project **pivoted** from its original scope. It began as a comparison of Federated Learning, Split Learning, and SplitFed on HAM10000 (see `thesis_proposal.tex` and `old/`). The current implementation focuses exclusively on **FL under controlled network conditions**, using CIFAR-10 for reproducibility and scalability.

---

## Features

- **Arbitrary N clients** — Configure any number of clients per experiment
- **Per-client network shaping** — Bandwidth, latency, and loss via `tc netem`
- **Non-IID data** — Dirichlet partitioning (α configurable) for realistic heterogeneity
- **Stats collection** — Per-round timing, bytes transferred, train/eval metrics
- **Experiment sweeps** — Automated bandwidth/latency/loss sweeps with checkpointing
- **Results export** — JSON per run, CSV aggregation for plotting

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop (or Docker Engine + Compose v2)
- macOS or Linux

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install pytest
```

### 2. Run tests (no Docker)

```bash
python3 -m pytest tests/unit/ -v
python3 -m pytest tests/integration/ -v
```

### 3. Smoke test in containers

```bash
docker compose -f infra/compose.yml build
docker compose -f infra/compose.yml up
```

Starts the server + 2 clients with synthetic data. Completes in ~10–15 seconds.

### 4. Run a real experiment (CIFAR-10, network shaping)

```bash
python infra/scripts/run_experiment.py experiments/heterogeneous.yaml
```

Generates the compose file, builds, and runs. Results are written to `results/<experiment_name>/`.

---

## Usage Guide

For detailed instructions—expected logs, network validation, manual compose generation, experiment config format—see **[docs/runbook.md](docs/runbook.md)**.

### Experiment config format

```yaml
experiment:
  name: "my-experiment"
  rounds: 10
  num_clients: 3
  alpha: 0.5   # Dirichlet: lower = more non-IID

server:
  port: 8000

clients:
  - id: 1
    network:
      bandwidth: "100mbit"
      latency: "5ms"
      loss: "0%"
  - id: 2
    network:
      bandwidth: "5mbit"
      latency: "100ms"
      loss: "1%"
  - id: 3
    network: {}   # no shaping
```

### Full QoS sweep

Run the full experiment matrix (bandwidth, latency, loss sweeps):

```bash
python infra/scripts/run_sweep.py              # run (resumes from checkpoints)
python infra/scripts/run_sweep.py --dry-run     # print plan only
python infra/scripts/run_sweep.py --only bw     # bandwidth sweep only
python infra/scripts/run_sweep.py --force       # ignore checkpoints, re-run all
```

Results go to `results/<experiment>/` with `.done` checkpoints for idempotent re-runs.

### Aggregate results for plotting

```bash
python infra/scripts/aggregate_results.py
```

Produces `results/client_stats.csv` and `results/server_stats.csv`.

---

## Repository Layout

```
src/                     # Python package
  models/                #   ResNet-18
  data/                  #   CIFAR-10, Dirichlet partitioning
  fl/                    #   FL core (aggregation, client, server)
  transport/             #   HTTP adapter
  stats/                 #   Per-round stats collector
  server_main.py         #   Server entrypoint
  client_main.py         #   Client entrypoint
infra/
  Dockerfile.ml-base     #   PyTorch + network tools image
  compose.yml            #   Smoke test (2 clients, synthetic)
  compose.generated.yml  #   Generated from experiment YAML
  scripts/
    run_experiment.py    #   One-command experiment runner
    run_sweep.py         #   Full QoS sweep
    generate_compose.py  #   YAML → compose
    aggregate_results.py #   JSON → CSV
    entrypoint.sh        #   tc/netem shaping before Python
experiments/             # YAML configs
  baseline.yaml          #   No shaping
  heterogeneous.yaml     #   Mixed network conditions
  qos-sweep-bw.yaml      #   Bandwidth sweep template
  scale-5.yaml, scale-10.yaml
tests/                   # Unit + integration
docs/
  runbook.md             #   Step-by-step usage
  architecture.md        #   Design overview
  future-work.md         #   Thesis future work
old/                     # Original SplitFed/HAM10000 reference code
```

---

## Data and Model

| Component | Details |
|-----------|---------|
| **Dataset** | CIFAR-10 (10 classes). Synthetic random data for smoke tests (`--synthetic`). |
| **Model** | ResNet-18 |
| **Partitioning** | Dirichlet(α) across clients; α=0.5 gives moderately non-IID. |

---

## Roadmap

- **Phase 1** ✓ — Container foundation, FL core, HTTP transport, network validation
- **Phase 2** ✓ — N-client orchestration, experiment YAML, per-client stats
- **Phase 3** ✓ — Experiment sweeps, JSON/CSV export, CIFAR-10 integration

See [docs/future-work.md](docs/future-work.md) for thesis-oriented extensions (bursty loss, bufferbloat, time-varying profiles, etc.).

---

## Reference

- **Original SplitFed codebase** (Thapa et al. 2022): `old/`
- **Thesis proposal** (FL, SL, SplitFed comparison): `thesis_proposal.tex`
- **Architecture**: [docs/architecture.md](docs/architecture.md)
