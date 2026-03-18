#!/bin/bash
set -e

if [ -n "${TC_BANDWIDTH}" ] || [ -n "${TC_LATENCY}" ] || [ -n "${TC_LOSS}" ]; then
    _bw="${TC_BANDWIDTH:-1000mbit}"
    _lat="${TC_LATENCY:-0ms}"
    _loss="${TC_LOSS:-0%}"

    echo "[ENTRYPOINT] Applying network shaping: bandwidth=${_bw} latency=${_lat} loss=${_loss}"

    echo "[ENTRYPOINT] tc qdisc add dev eth0 root netem delay ${_lat} rate ${_bw} loss ${_loss}"
    tc qdisc add dev eth0 root netem delay "${_lat}" rate "${_bw}" loss "${_loss}"

    echo "[ENTRYPOINT] Network shaping applied (egress/upload only)."
else
    echo "[ENTRYPOINT] No network shaping configured. Starting directly."
fi

echo "[ENTRYPOINT] Starting: $*"
exec "$@"
