"""Network interface I/O byte counters.

Reads raw tx/rx counters from the kernel's sysfs interface.  On Linux these
counters include every byte that actually leaves or enters the NIC — TCP
retransmissions, ACKs, and all lower-layer framing — so comparing them
against application-layer payload sizes directly quantifies retransmission
overhead under packet-loss conditions.

On non-Linux machines (e.g. macOS dev boxes) the sysfs paths don't exist and
the function silently returns (0, 0) so callers can treat the metric as
unavailable without any special-casing.
"""


def read_interface_bytes(iface: str = "eth0") -> tuple[int, int]:
    """Read (tx_bytes, rx_bytes) from /sys/class/net/{iface}/statistics/.

    Returns (0, 0) if the interface or files don't exist (e.g. on macOS dev
    machines).
    """
    try:
        tx = int(open(f"/sys/class/net/{iface}/statistics/tx_bytes").read().strip())
        rx = int(open(f"/sys/class/net/{iface}/statistics/rx_bytes").read().strip())
        return tx, rx
    except (FileNotFoundError, ValueError):
        return 0, 0
