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

Each container will capture bytes sent/received, per-round wall-clock time, and throughput. These are collected after training and output as a report (CSV or JSON).

## Data and Model

- **Dataset**: HAM10000 skin lesion classification. Synthetic data available for smoke tests.
- **Model**: ResNet18, 7 classes.
- **Partitioning**: IID across clients.

## File Layout

- `src/` — FL core, data, model, transport interfaces, HTTP adapter.
- `infra/` — Dockerfiles, Compose, network shaping and validation scripts.
- `tests/` — Unit and integration tests.
- `docs/` — This documentation set.

## Reference

Original thesis proposal (covering FL, SL, SplitFed) retained at [thesis_proposal.tex](../thesis_proposal.tex) for academic context. This implementation focuses exclusively on FL.
