# Decision Log (ADR-style)

Decisions that affect implementation. Each entry is traceable and reversible later if needed.

---

## ADR-1: Transport — HTTP + JSON first

**Context**: Need a concrete protocol for server-client communication.

**Decision**: HTTP+JSON via FastAPI (server) and httpx (client). Model state serialized as binary blobs with `torch.save`.

**Consequences**: Easy to debug; payload size for full model state is large but acceptable. Can optimize later (binary body, gRPC).

---

## ADR-2: Transport-agnostic core

**Context**: Protocol might change for experiments or optimization.

**Decision**: FL logic (aggregation, training, model distribution) does not depend on HTTP. Abstract interfaces (`get_global_model`, `submit_update`) with adapter implementations.

**Consequences**: Can swap HTTP for gRPC/sockets without rewriting FL logic.

---

## ADR-3: Network shaping inside containers (NET_ADMIN)

**Context**: Need `tc`/`netem` for bandwidth and latency shaping.

**Decision**: Apply inside containers with `NET_ADMIN` capability. Scripts run inside to add/change qdiscs on eth0.

**Consequences**: Simpler Compose workflow; no host-side veth scripting.

---

## ADR-4: Scope — Federated Learning only

**Context**: Original thesis proposal covered FL, Split Learning, and SplitFed. Thesis has been simplified.

**Decision**: Implement only Federated Learning. The simulator supports arbitrary client count, configurable network conditions, and network statistics reporting. SL and SplitFed are out of scope.

**Consequences**: Focused implementation; the reference SplitFed codebase is archived in `old/` for academic context only.

---

## ADR-5: Polling-based round synchronization

**Context**: The HTTP server needs to coordinate N clients per round.

**Decision**: Clients POST their update then poll `GET /round_status` until all clients have reported, then fetch the new global model. No blocking waits in request handlers.

**Consequences**: Simple, deadlock-free. Slight polling overhead but negligible for FL round durations.
