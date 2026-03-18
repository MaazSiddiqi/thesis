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

## 8. Run an Experiment (Phase 2)

Phase 2 adds per-client network shaping and a stats summary. Instead of the hand-written `compose.yml`, experiments are driven by a YAML config.

### One-command run

```bash
python infra/scripts/run_experiment.py experiments/heterogeneous.yaml
```

This generates `infra/compose.generated.yml`, builds the image (if needed), and runs all containers. When the clients finish you will see a stats table in each client's logs:

```
[CLIENT 2] ── Network Stats Summary ────────────────────────────────────────
[CLIENT 2]  round  get_model_ms  submit_ms  train_ms  submit_bytes  poll_count
[CLIENT 2]      0          32.1      890.2    3210.0      44040012           3
[CLIENT 2]      1          28.4      820.9    3198.3      44040012           2
[CLIENT 2] ────────────────────────────────────────────────────────────────
```

### Expected logs (Phase 2, heterogeneous example)

The exact numbers will vary, but the sequence should look roughly like:

**Server** (after all rounds complete):

```text
fl-server  | [SERVER] Experiment=heterogeneous Starting FL server on 0.0.0.0:8000 (expecting 3 clients, 3 rounds)
fl-server  | [SERVER] Round 0: received update from client 1 (1/3)
fl-server  | [SERVER] Round 0: received update from client 2 (2/3)
fl-server  | [SERVER] Round 0: received update from client 3 (3/3)
fl-server  | [SERVER] Round 0: aggregating 3 updates (round took 3120.0 ms)
...
fl-server  | [SERVER] Experiment=heterogeneous Summary (clients=3, rounds=3)
fl-server  | [SERVER] ── Server Round Stats Summary ───────────────
fl-server  | [SERVER]  round    duration_ms
fl-server  | [SERVER]      0         3120.0
fl-server  | [SERVER]      1         4285.3
fl-server  | [SERVER]      2         3978.6
fl-server  | [SERVER] ─────────────────────────────────────────────
fl-server  | [SERVER] All 3 rounds complete. Shutting down.
```

**One client** (with constrained bandwidth, showing its config and network stats):

```text
fl-client-2 | [CLIENT 2] Experiment=heterogeneous Client 2 starting, server=http://fl-server:8000, rounds=3
fl-client-2 | [CLIENT 2] Round 0: fetching global model
fl-client-2 | [CLIENT 2] Round 0: training locally
...
fl-client-2 | [CLIENT 2] Experiment=heterogeneous Client 2 finished 3 rounds
fl-client-2 | [CLIENT 2] Experiment=heterogeneous Client 2 config: rounds=3, bandwidth=5mbit, latency=100ms, loss=1%
fl-client-2 | [CLIENT 2] ── Network Stats Summary ────────────────────────────────────────
fl-client-2 | [CLIENT 2]   round  get_model_ms  submit_ms  train_ms  submit_bytes  poll_count
fl-client-2 | [CLIENT 2]       0          32.1      890.2    3210.0      44040012           3
fl-client-2 | [CLIENT 2]       1          28.4      820.9    3198.3      44040012           2
fl-client-2 | [CLIENT 2]       2          30.7      945.5    3205.1      44040012           4
fl-client-2 | [CLIENT 2] ────────────────────────────────────────────────────────────────
```

### Manual steps (if you prefer)

```bash
# Step 1: generate compose file from config
python infra/scripts/generate_compose.py experiments/heterogeneous.yaml
# → infra/compose.generated.yml

# Step 2: build (first time or after requirements change)
docker compose -f infra/compose.generated.yml build

# Step 3: run
docker compose -f infra/compose.generated.yml up
```

### Experiment config format

```yaml
experiment:
  name: "heterogeneous"
  rounds: 3

server:
  port: 8000

clients:
  - id: 1
    network:
      bandwidth: "10mbit"
      latency: "10ms"
      loss: "0%"
  - id: 2
    network:
      bandwidth: "1mbit"
      latency: "100ms"
      loss: "1%"
  - id: 3
    network: {}          # no shaping
```

Omit any network key (or use `network: {}`) to leave that parameter unshapped. See `experiments/baseline.yaml` for a no-shaping reference run.

### How network shaping is applied

The `infra/scripts/entrypoint.sh` script runs before Python inside each client container. If `TC_BANDWIDTH`, `TC_LATENCY`, or `TC_LOSS` env vars are present (injected by the compose generator), it runs:

```bash
tc qdisc add dev eth0 root netem delay <latency> rate <bandwidth> loss <loss>
```

then hands off to the Python process via `exec "$@"`.

### Cleanup

```bash
docker compose -f infra/compose.generated.yml down
```

---

## Known Limitations

- Smoke tests use synthetic data; real dataset training not yet wired to containers.
- Stats are logged to terminal only; CSV/JSON export is Phase 3.
- `signal.pause()` in server_main.py is Unix-only.

## TODO: Migrate to CIFAR-10 for Thesis Experiments

Before running the accuracy sweep experiments for the thesis, the simulator needs
to be migrated from synthetic data to CIFAR-10:

1. Add CIFAR-10 download + preprocessing to the client data pipeline.
2. Update model output classes from 7 to 10 in the server config.
3. Implement dataset partitioning across clients (IID split to start).
4. Verify a full training run converges to a reasonable accuracy at baseline
   network conditions (100 Mbps, 5 ms latency, 0% loss) before starting the sweep.

## What's Next (Phase 3)

- Export stats to CSV/JSON files for cross-experiment analysis.
- Run an experiment matrix (e.g. vary bandwidth 1–100 Mbit, measure round time).
