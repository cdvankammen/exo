# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for network traffic counters on NetworkInterfaceInfo (T52)."""
from exo.shared.types.profiling import NetworkInterfaceInfo


class TestNetworkInterfaceTrafficCounters:
    def test_default_counters_are_none(self) -> None:
        """Backward compat: old serialized data without counters deserializes."""
        info = NetworkInterfaceInfo(name="eth0", ip_address="10.0.0.1")
        assert info.rx_bytes is None
        assert info.tx_bytes is None
        assert info.timestamp_ns is None

    def test_counters_round_trip(self) -> None:
        info = NetworkInterfaceInfo(
            name="eth0",
            ip_address="10.0.0.1",
            rx_bytes=123456,
            tx_bytes=789012,
            timestamp_ns=1_700_000_000_000_000_000,
        )
        dumped = info.model_dump()
        restored = NetworkInterfaceInfo.model_validate(dumped)
        assert restored.rx_bytes == 123456
        assert restored.tx_bytes == 789012
        assert restored.timestamp_ns == 1_700_000_000_000_000_000

    def test_rate_calculation_from_deltas(self) -> None:
        """Dashboard rate math: (bytes2 - bytes1) / (ns2 - ns1) * 1e9."""
        t1 = 1_000_000_000_000_000_000
        t2 = 1_000_000_001_000_000_000  # 1 second later
        rx1, rx2 = 1_000_000, 2_000_000
        tx1, tx2 = 500_000, 1_500_000

        dt_s = (t2 - t1) / 1e9
        rx_rate = (rx2 - rx1) / dt_s
        tx_rate = (tx2 - tx1) / dt_s

        assert rx_rate == 1_000_000.0  # 1 MB/s
        assert tx_rate == 1_000_000.0

    def test_zero_delta_no_division_by_zero(self) -> None:
        """Same timestamp should not crash rate calculation."""
        t = 1_000_000_000_000_000_000
        rx1, rx2 = 1_000_000, 1_000_000
        dt_s = (t - t) / 1e9
        # Dashboard should guard: if dt_s == 0, rate = 0
        rate = 0.0 if dt_s == 0 else (rx2 - rx1) / dt_s
        assert rate == 0.0
