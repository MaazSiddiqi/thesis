# Network Validation Protocol

Checks to confirm that Linux network utilities and container networking work as required. All steps are scriptable with PASS/FAIL output.

## Utilities Required

| Utility | Purpose | Package |
|---------|---------|---------|
| `tc` | Traffic control (qdisc, bandwidth, netem) | `iproute2` |
| `netem` | Latency, jitter, loss | Kernel module `sch_netem` |
| `iperf3` | Throughput measurement and competing traffic | `iperf3` |
| `ping` | Latency / reachability | `iputils-ping` |

Containers need these installed and `NET_ADMIN` capability for `tc`/`netem`.

## Where Shaping Is Applied

Inside the container on `eth0`. Scripts run after container start and execute `tc qdisc add/replace ...`.

## Checks

### 1. Tool availability

Run `tc qdisc show`, `ping -c 1 127.0.0.1`, `iperf3 --version` in each container. All must exit 0.

### 2. Container-to-container connectivity

From client container: resolve server hostname and `ping -c 3 <server>`. Replies received.

### 3. TCP reachability

`iperf3 -s` in one container; `iperf3 -c <server> -t 2` from another. Connection established, throughput reported.

### 4. Latency shaping

- Baseline: `ping -c 5 <peer>`, record avg RTT.
- Apply: `tc qdisc add dev eth0 root netem delay 50ms`.
- Shaped: `ping -c 5 <peer>`, record avg RTT.
- Cleanup: `tc qdisc del dev eth0 root`.
- Expected: shaped RTT ~ baseline + 50ms (within ±20%).

### 5. Bandwidth shaping

Cap egress to known value (e.g. 10Mbit) with `tc`. Measure with `iperf3`. Observed throughput at or below limit.

## Single-Command Smoke

`./infra/scripts/validate_network.sh` — brings up stack, runs all checks, reports PASS/FAIL. Exit 0 = all pass.
