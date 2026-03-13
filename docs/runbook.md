# Runbook

Step-by-step to reproduce the current state from scratch.

## Prerequisites

- Python 3.10+ with pip
- Docker Desktop (or Docker Engine + Compose v2)
- macOS or Linux host

## 1. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install pytest
```

## 2. Run Unit Tests

```bash
python3 -m pytest tests/unit/ -v
```

Expected: 12 tests pass (aggregation, model, partitioning, serialization).

## 3. Run Integration Test

```bash
python3 -m pytest tests/integration/ -v
```

Expected: 1 test passes in under 10 seconds (full FL round over HTTP on localhost).

## 4. Build Docker Images

```bash
docker compose -f infra/compose.yml build
```

## 5. Run FL Smoke Test in Containers

```bash
docker compose -f infra/compose.yml up
```

Starts `fl-server` + 2 clients using synthetic (random) data. The full run completes in roughly 10–15 seconds on a modern laptop.

### Expected log output

The logs below are from a verified run (2026-03-13). Timestamps and loss/accuracy values will vary, but the structure and sequence must match.

**Server startup** (first ~6 seconds while healthcheck warms up):
```
fl-server | [SERVER] Building ResNet18 with 7 classes
fl-server | [SERVER] Starting FL server on 0.0.0.0:8000 (expecting 2 clients)
fl-server | [SERVER] HTTP server starting on 0.0.0.0:8000
```

**Both clients start simultaneously** and fetch the initial global model:
```
fl-client-1 | [CLIENT 1] Client 1 starting, server=http://fl-server:8000, rounds=1
fl-client-2 | [CLIENT 2] Client 2 starting, server=http://fl-server:8000, rounds=1
fl-client-1 | HTTP Request: GET http://fl-server:8000/health "HTTP/1.1 200 OK"
fl-client-2 | HTTP Request: GET http://fl-server:8000/health "HTTP/1.1 200 OK"
fl-client-1 | [CLIENT 1] Round 0: fetching global model
fl-client-2 | [CLIENT 2] Round 0: fetching global model
fl-client-1 | HTTP Request: GET http://fl-server:8000/model "HTTP/1.1 200 OK"
fl-client-2 | HTTP Request: GET http://fl-server:8000/model "HTTP/1.1 200 OK"
```

**Both clients train in parallel** (each on their own private synthetic data, ~3s on CPU):
```
fl-client-1 | [CLIENT 1] Round 0: training locally
fl-client-2 | [CLIENT 2] Round 0: training locally
fl-client-1 | [CLIENT 1] Round 0: train loss=2.1455 acc=7.81%
fl-client-2 | [CLIENT 2] Round 0: train loss=2.0388 acc=14.06%
```
Train accuracy is near-random (~7–14%) because the model is randomly initialised and only runs 1 epoch on 64 random images. This is expected.

**Clients submit updates; server waits for both before aggregating:**
```
fl-client-1 | HTTP Request: POST http://fl-server:8000/update/1 "HTTP/1.1 200 OK"
fl-server   | [SERVER] Round 0: received update from client 1 (1/2)
fl-client-2 | HTTP Request: POST http://fl-server:8000/update/2 "HTTP/1.1 200 OK"
fl-server   | [SERVER] Round 0: received update from client 2 (2/2)
fl-server   | [SERVER] Round 0: aggregating 2 updates
```
The server only logs "aggregating" after the second update arrives. The order of client 1 vs client 2 may swap between runs.

**Clients poll for round completion, download averaged model, evaluate:**
```
fl-client-1 | HTTP Request: GET http://fl-server:8000/round_status "HTTP/1.1 200 OK"
fl-client-2 | HTTP Request: GET http://fl-server:8000/round_status "HTTP/1.1 200 OK"
fl-client-1 | HTTP Request: GET http://fl-server:8000/model "HTTP/1.1 200 OK"
fl-client-2 | HTTP Request: GET http://fl-server:8000/model "HTTP/1.1 200 OK"
fl-client-1 | [CLIENT 1] Round 0: eval  loss=1.8778 acc=21.88%
fl-client-2 | [CLIENT 2] Round 0: eval  loss=1.8672 acc=35.94%
fl-client-1 | [CLIENT 1] Client 1 finished 1 rounds
fl-client-2 | [CLIENT 2] Client 2 finished 1 rounds
```
After FedAvg, both clients load the **same** averaged model. Their eval accuracies differ only because they evaluate on different random batches — this is correct behaviour.

### Why eval accuracy is higher than train accuracy

Both clients start from a random model (≈14% on 7 classes). After one local training step their *individual* models may not generalise. But after FedAvg the averaged model benefits from both clients' gradient directions, which typically produces a slightly better generalisation than either client alone — hence eval > train here.

### Verifying via curl

With the server still running, from your host machine:

```bash
curl http://localhost:8000/health
# → {"status":"ok"}

curl http://localhost:8000/round_status
# → {"round":1,"updates_received":2,"updates_needed":2,"complete":true}
```

## 6. Network Validation

```bash
docker compose -f infra/compose.yml --profile network-test up -d
chmod +x infra/scripts/validate_connectivity.sh infra/scripts/validate_network.sh
./infra/scripts/validate_connectivity.sh
./infra/scripts/validate_network.sh
```

Expected: both scripts report PASSED.

## 7. Cleanup

```bash
docker compose -f infra/compose.yml --profile network-test down
docker compose -f infra/compose.yml down
```

## Known Limitations

- Smoke tests use synthetic data; full HAM10000 training not yet wired to containers.
- Fixed at 2 clients in Compose; arbitrary N-client orchestration is Phase 2.
- No network statistics collection yet (Phase 2).
- `signal.pause()` in server_main.py is Unix-only.

## What's Next (Phase 2)

- Arbitrary N-client Compose generation or dynamic scaling.
- Experiment configuration file (network params, client count, rounds).
- Network statistics collection and reporting (bytes transferred, wall-clock time, throughput per round).
