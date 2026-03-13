#!/usr/bin/env bash
# Check presence of tc, netem, ping, iperf3 inside a container.
# Usage: docker exec <container> /app/infra/scripts/check_tools.sh
# Or: docker exec <container> bash -c 'tc qdisc show; ping -c 1 127.0.0.1; iperf3 --version'
# This script is intended to be run inside a container (e.g. fl-client-1) that has /app mounted.

set -e
echo "Checking tc..."
tc qdisc show || { echo "tc failed"; exit 1; }
echo "Checking ping..."
ping -c 1 127.0.0.1 || { echo "ping failed"; exit 1; }
echo "Checking iperf3..."
iperf3 --version || { echo "iperf3 failed"; exit 1; }
echo "All tools OK"
