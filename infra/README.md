# Phase 01 Infrastructure

- **compose.yml** — FL server, two FL clients, iperf3 server. All on network `phase1-fl-net` with `NET_ADMIN` for tc/netem.
- **Dockerfile.ml-base** — Python 3.10 + PyTorch deps + iproute2, ping, iperf3. Used for `fl-server` and `fl-client-*`.
- **Dockerfile.network-test** — Minimal image (Debian + network tools only) for fast validation.
- **scripts/validate_connectivity.sh** — One-shot: bring up stack, ping fl-server from client, iperf3 client→iperf-server, check tools. Run from repo root: `./infra/scripts/validate_connectivity.sh`.
- **scripts/check_tools.sh** — Run inside a container to verify tc/ping/iperf3.

Build from repo root:

```bash
docker compose -f infra/compose.yml build
docker compose -f infra/compose.yml up -d
./infra/scripts/validate_connectivity.sh
```

See [docs/phase-01/network-validation.md](../docs/phase-01/network-validation.md) for the full validation protocol.
