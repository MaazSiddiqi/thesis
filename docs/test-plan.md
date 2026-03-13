# Test Plan

## Test Levels

| Level | Scope | When |
|-------|-------|------|
| Unit | Single modules (FedAvg, partitioning, model, serialization) | On change |
| Integration | Server + client on localhost over HTTP | Before merging transport changes |
| Smoke / E2E | Full stack in containers; at least one FL round | Before release |

## Unit Tests

| Component | What to test | Location |
|-----------|-------------|----------|
| FedAvg | Average correctness for 2+ state dicts | `tests/unit/test_aggregation.py` |
| IID partition | Sizes, disjoint, coverage | `tests/unit/test_partitioning.py` |
| ResNet18 | Forward pass shape | `tests/unit/test_model.py` |
| Serialization | state_dict round-trip | `tests/unit/test_serialization.py` |

## Integration Tests

| Scenario | What to test | Location |
|----------|-------------|----------|
| One FL round (localhost) | 2 clients complete round, server aggregates | `tests/integration/test_fl_round.py` |

## Smoke / E2E

| Scenario | How |
|----------|-----|
| Containers up | `docker compose up`, check healthcheck |
| Network validation | `./infra/scripts/validate_network.sh` |
| FL round in containers | Compose up, clients log "finished", server aggregates |

## Commands

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
./infra/scripts/validate_network.sh
docker compose -f infra/compose.yml up
```

## File-to-Doc Traceability

| File | Justifying doc |
|------|---------------|
| `src/fl/aggregation.py` | architecture.md (FL Server) |
| `src/fl/client_core.py` | architecture.md (FL Client) |
| `src/fl/server_core.py` | architecture.md (FL Server) |
| `src/transport/interface.py` | architecture.md (Transport); decision-log.md ADR-2 |
| `src/transport/http_adapter.py` | decision-log.md ADR-1, ADR-5 |
| `src/data/partitioning.py` | architecture.md (Data and Model) |
| `src/models/resnet.py` | architecture.md (Data and Model) |
| `infra/Dockerfile.*` | architecture.md (Network Shaping) |
| `infra/compose.yml` | architecture.md (Component Boundaries) |
| `infra/scripts/validate_network.sh` | network-validation.md |
