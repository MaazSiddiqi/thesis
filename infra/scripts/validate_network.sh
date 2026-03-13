#!/usr/bin/env bash
# Phase 01: End-to-end network validation (tc/netem + ping/iperf3).
# See docs/phase-01/network-validation.md
# Run from repo root: ./infra/scripts/validate_network.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/infra/compose.yml"
cd "$REPO_ROOT"

echo "=== Phase 01 network validation (tc/netem + ping/iperf3) ==="

echo "Ensuring network-test stack is up..."
docker compose -f "$COMPOSE_FILE" --profile network-test up -d fl-server-idle fl-client-idle iperf-server
sleep 5

CLIENT="fl-client-idle"
PEER="fl-server-idle"

echo "--- 1) Tool availability inside $CLIENT ---"
docker exec "$CLIENT" tc qdisc show >/dev/null 2>&1 || true
docker exec "$CLIENT" tc -Version >/dev/null 2>&1 || { echo "FAIL: tc missing"; exit 1; }
docker exec "$CLIENT" ping -c 1 127.0.0.1 >/dev/null 2>&1 || { echo "FAIL: ping missing"; exit 1; }
docker exec "$CLIENT" iperf3 --version >/dev/null 2>&1 || { echo "FAIL: iperf3 missing"; exit 1; }
echo "PASS: tools present"

echo "--- 2) Baseline ping RTT from $CLIENT to $PEER ---"
BASELINE_RTT_RAW="$(docker exec "$CLIENT" ping -c 5 "$PEER" | tail -1 | awk '{print $4}' | cut -d'/' -f2 || echo "")"
if [ -z "$BASELINE_RTT_RAW" ]; then
  echo "FAIL: could not parse baseline RTT"; exit 1;
fi
echo "Baseline avg RTT: ${BASELINE_RTT_RAW} ms"

echo "--- 3) Apply netem delay 50ms on eth0 inside $CLIENT ---"
docker exec "$CLIENT" tc qdisc del dev eth0 root 2>/dev/null || true
docker exec "$CLIENT" tc qdisc add dev eth0 root netem delay 50ms || { echo "FAIL: tc netem add"; exit 1; }
docker exec "$CLIENT" tc qdisc show dev eth0 || true

echo "--- 4) Shaped ping RTT from $CLIENT to $PEER ---"
SHAPED_RTT_RAW="$(docker exec "$CLIENT" ping -c 5 "$PEER" | tail -1 | awk '{print $4}' | cut -d'/' -f2 || echo "")"
if [ -z "$SHAPED_RTT_RAW" ]; then
  echo "FAIL: could not parse shaped RTT"; exit 1;
fi
echo "Shaped avg RTT: ${SHAPED_RTT_RAW} ms"

echo "--- 5) Compare RTTs (expect shaped > baseline + ~40ms) ---"
python3 - "$BASELINE_RTT_RAW" "$SHAPED_RTT_RAW" << 'PY'
import sys
try:
    base = float(sys.argv[1])
    shaped = float(sys.argv[2])
except Exception:
    print("FAIL: invalid RTT values")
    sys.exit(1)

delta = shaped - base
print(f"delta RTT = {delta:.2f} ms")
if delta < 30.0:
    print("FAIL: shaped RTT did not increase enough")
    sys.exit(1)
print("PASS: shaped RTT increased as expected")
PY

echo "--- 6) Cleanup netem on $CLIENT ---"
docker exec "$CLIENT" tc qdisc del dev eth0 root 2>/dev/null || true

echo "=== Phase 01 network validation PASSED ==="
