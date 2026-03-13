#!/usr/bin/env bash
# Phase 01: Prove container discovery and communication.
# See docs/phase-01/network-validation.md and test-plan.md
# Run from repo root: ./infra/scripts/validate_connectivity.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/infra/compose.yml"
cd "$REPO_ROOT"

echo "=== Phase 01 connectivity validation ==="

echo "Ensuring network-test stack is up..."
docker compose -f "$COMPOSE_FILE" --profile network-test up -d fl-server-idle fl-client-idle iperf-server
sleep 5

SERVER="fl-server-idle"
CLIENT="fl-client-idle"

echo "--- 1) Resolve and ping $SERVER from $CLIENT ---"
docker exec "$CLIENT" ping -c 3 "$SERVER" || { echo "FAIL: ping"; exit 1; }
echo "PASS: ping"

echo "--- 2) iperf3 throughput ($CLIENT -> iperf-server) ---"
docker exec "$CLIENT" iperf3 -c iperf-server -t 2 || { echo "FAIL: iperf3"; exit 1; }
echo "PASS: iperf3"

echo "--- 3) Tool availability in $CLIENT ---"
docker exec "$CLIENT" tc -Version 2>/dev/null || docker exec "$CLIENT" tc qdisc show 2>/dev/null || { echo "FAIL: tc"; exit 1; }
docker exec "$CLIENT" iperf3 --version 2>/dev/null || { echo "FAIL: iperf3 version"; exit 1; }
echo "PASS: tools available"

echo "=== All connectivity checks PASSED ==="
