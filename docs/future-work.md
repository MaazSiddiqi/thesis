## Future Work: Network-Aware FL Simulator

This project currently supports:
- Running an arbitrary number of FL clients in Docker containers.
- Per-client network shaping (bandwidth, latency, loss) via `tc netem`.
- In-memory collection of per-round client stats and logging to stdout.

Below are concrete directions to deepen the network analysis. These are intentionally framed at a high level so they can be used directly in the thesis “Future Work” section.

### 1. Richer Network Impairment Models

- **Bursty loss and jitter**: Today we model constant loss and delay. Using additional `tc netem` parameters (e.g., bursty loss and jitter) would better approximate real Wi‑Fi or cellular links, where the connection is mostly fine but occasionally very bad for short periods. This would allow experiments on how short outages or delay spikes on a single client stretch global round times or trigger timeouts.

- **Packet reordering and duplication**: Extending the netem configuration to include reordering and occasional packet duplication would stress-test the robustness of the HTTP/TCP stack underneath FL. While TCP hides most of this from the application, measuring the resulting changes in submit time and retransmissions would provide insight into how sensitive FL is to path “weirdness”.

### 2. Queueing and Bufferbloat Effects

- **Queue disciplines and shaping**: The current setup mainly limits bandwidth. Future work could introduce explicit queueing disciplines (e.g., `tbf`, `fq_codel`) to study how different router behaviors affect FL. This would model scenarios like home routers with very large buffers (bufferbloat) versus smarter queue management, and quantify their impact on round duration and client timeouts.

- **Symmetric and asymmetric links**: The current implementation only shapes egress (upload) traffic because the IFB kernel module required for ingress shaping is unavailable on macOS Docker Desktop. On a native Linux host, IFB is available and would enable independent downlink and uplink shaping. Future experiments could independently control download (server→client model broadcast) and upload (client→server update) bandwidth to study which direction dominates FL performance under realistic asymmetric conditions (e.g., home broadband).

### 3. Time-Varying Network Profiles

- **Mobility and congestion dynamics**: In the current design, each client’s network profile is static over the lifetime of an experiment. A natural extension is to schedule profile changes over time (for example, “good” for the first few rounds, then “poor”, then “recovered”). This would emulate client mobility or changing congestion and allow analysis of how transient degradations affect convergence time and timeout rates.

- **Round-aware shaping**: Another variant is to change network conditions at specific round boundaries (e.g., intentionally degrading one client during round \(k\)). This would enable controlled studies of the sensitivity of FL to when in the training process a client becomes the bottleneck.

### 4. Cross-Traffic and Shared Links

- **Background flows**: Currently, each client effectively has a dedicated shaped link. Introducing synthetic cross-traffic (for example, via `iperf3` or simple HTTP downloads in sidecar containers) would simulate other applications sharing the same access network. Comparing “dedicated” versus “shared” scenarios at the same nominal bandwidth would show how contention changes effective throughput and round completion times.

- **Application-level contention scenarios**: Future work could design scenarios where some clients experience cross-traffic only during specific rounds, or only during upload, to more closely mimic user behavior (e.g., streaming video while training).

### 5. Deeper Transport and OS-Level Metrics

- **Interface byte and packet counters**: The current stats focus on logical payload bytes and timing at the application layer. Reading interface counters from the OS (e.g., `ip -s link show eth0` or `/sys/class/net/eth0/statistics/*`) before and after each round would quantify the true bytes on the wire, including protocol overhead and retransmissions. This can reveal cases where lossy conditions cause significantly more traffic than the model size alone suggests.

- **TCP health snapshots**: Periodically sampling TCP connection state (for example via `ss -ti`) would expose metrics like RTT, RTT variance, congestion window size, and retransmission counts. Correlating these with per-round submit times would help explain *why* certain network profiles produce longer uploads or higher timeout rates at a transport level.

### 6. Packet-Level Tracing for Representative Scenarios

- **Pcap-based deep dives**: For a small number of representative experiments, packet captures (e.g., via `tcpdump`/`tshark` and analysis in Wireshark) could be used to visualise throughput, RTT distributions, and retransmissions over time. These traces would serve as ground truth for validating that the configured network impairments behave as intended and provide rich material for figures in the thesis.

- **Validation of shaping policies**: Packet-level analysis could also be used to verify that queueing and shaping policies (e.g., intended bandwidth caps or latency targets) are being met, and to adjust `tc` parameters where the observed behaviour diverges from the target model.

### 7. Extended Client and Server Metrics

- **Finer-grained timing breakdowns**: The existing client stats already track `get_model_ms`, `submit_ms`, and `train_ms`. Future extensions could add explicit metrics for time spent waiting for the server to advance the round and overall end-to-end round duration per client. This would help separate server-side delays from network and compute effects.

- **Aggregated server-side statistics**: On the server, additional summaries could include per-round distributions of client submit times, counts of timed-out or failed clients, and correlations between client network profiles and their effective contribution to each round. This would enable more detailed analysis of fairness and straggler impact under heterogeneous conditions.

