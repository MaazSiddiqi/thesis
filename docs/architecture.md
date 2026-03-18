# Architecture: Network-Aware FL Simulator

## Goal

A Docker-based Federated Learning simulator where:
- An arbitrary number of clients train in parallel.
- Network conditions (bandwidth, latency, congestion) are configurable per-experiment.
- Network statistics (bytes transferred, elapsed time, throughput) are collected and reported.

## Component Boundaries

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Experiment Config                                │
│  (client count, rounds, network params) → Compose orchestration         │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  FL Server    │    │  FL Client 1  │    │  FL Client N  │
│  (container)  │◄──►│  (container)  │    │  (container)  │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        │    Transport abstraction (interface)    │
        │    - get_global_model()                 │
        │    - submit_update(state_dict)          │
        └────────────────────────────────────────┘
                             │
                    tc / netem shaping
                    (inside containers)
```

### FL Server

- Holds global model; aggregates client updates (FedAvg); distributes new global model each round.
- Transport-agnostic: speaks through abstract interfaces, not HTTP directly.

### FL Client

- Receives global model; trains locally; sends updated weights back to server.
- Transport-agnostic: same interface abstraction.

### Transport Layer

Abstract interfaces (`ServerTransport`, `ClientTransport`) with a concrete HTTP+JSON adapter. The adapter can be swapped without changing FL logic.

### Network Shaping

Applied inside containers via `tc`/`netem` (requires `NET_ADMIN` capability). Scripts apply bandwidth limits, latency, and jitter before training starts. See [network-validation.md](network-validation.md).

### Network Statistics (Phase 2)

Each client container captures bytes sent/received and wall-clock time per HTTP call via `StatsCollector` (`src/stats/collector.py`). After all rounds complete the client logs a summary table to stdout:

```
── Network Stats Summary ────────────────────────────────────────────────
 round  get_model_ms  submit_ms  train_ms  submit_bytes  poll_count
     0          32.1      410.5    3210.0      44040012           1
     1          28.4      890.2    3198.3      44040012           3
─────────────────────────────────────────────────────────────────────────
```

High `submit_ms` on a client with constrained bandwidth directly quantifies the network's impact on FL round time.

Stats are kept in memory inside the client process and emitted only as a formatted table to stdout; no files are written in Phase 2.

The server also records per-round aggregation durations and prints a short server-side summary table after all configured rounds complete.

## Data and Model

- **Dataset**: CIFAR-10 (10 classes). Synthetic data available for smoke tests via `--synthetic`.
- **Model**: ResNet-18.
- **Partitioning**: Dirichlet(α) across clients; α configurable per experiment.

## File Layout

- `src/` — FL core, data, model, transport interfaces, HTTP adapter, stats collector.
- `infra/` — Dockerfiles, Compose, entrypoint script, network shaping and validation scripts.
- `experiments/` — YAML experiment configs (one file per experiment run).
- `tests/` — Unit and integration tests.
- `docs/` — This documentation set.

## Reference

Original thesis proposal (covering FL, SL, SplitFed) retained at [thesis_proposal.tex](../thesis_proposal.tex) for academic context. This implementation focuses exclusively on FL.
