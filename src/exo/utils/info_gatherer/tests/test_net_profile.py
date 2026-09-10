from unittest.mock import AsyncMock, MagicMock, patch

from exo.utils.info_gatherer.net_profile import (
    bandwidth_changed_materially,
    latency_changed_materially,
    probe_bandwidth,
)


def test_first_measurement_is_material():
    assert latency_changed_materially(None, 0.5)


def test_small_absolute_jitter_is_ignored():
    # sub-noise-floor changes never republish, even when the factor is large
    assert not latency_changed_materially(0.4, 1.9)
    assert not latency_changed_materially(1.9, 0.4)


def test_small_relative_change_is_ignored():
    assert not latency_changed_materially(40.0, 55.0)
    assert not latency_changed_materially(55.0, 40.0)


def test_large_change_is_material():
    # e.g. traffic silently rerouted from thunderbolt to a WAN path
    assert latency_changed_materially(0.8, 45.0)
    assert latency_changed_materially(45.0, 0.8)


def test_bandwidth_first_measurement_is_material():
    assert bandwidth_changed_materially(None, 100.0)


def test_bandwidth_small_relative_change_is_ignored():
    assert not bandwidth_changed_materially(100.0, 120.0)
    assert not bandwidth_changed_materially(120.0, 100.0)


def test_bandwidth_large_change_is_material():
    # e.g. link swapped from ethernet to thunderbolt (or vice versa)
    assert bandwidth_changed_materially(100.0, 45.0)
    assert bandwidth_changed_materially(45.0, 100.0)


async def test_probe_bandwidth_computes_mbps():
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.content = b"\0" * 1_048_576  # 1 MiB
    client.get = AsyncMock(return_value=response)

    # 1 MiB in 0.5s -> 2 MiB/s
    calls = {"n": 0}

    def _fake_perf_counter() -> float:
        calls["n"] += 1
        if calls["n"] == 1:
            return 1000.0
        return 1000.5

    with patch("time.perf_counter", side_effect=_fake_perf_counter):
        mbps = await probe_bandwidth("10.0.0.1", client, api_port=52415)

    assert mbps is not None
    assert abs(mbps - 2.0) < 0.01


async def test_probe_bandwidth_short_body_returns_none():
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.content = b"\0" * 100  # truncated
    client.get = AsyncMock(return_value=response)

    mbps = await probe_bandwidth("10.0.0.1", client, api_port=52415)

    assert mbps is None


async def test_probe_bandwidth_error_returns_none():
    import httpx

    client = MagicMock()
    client.get = AsyncMock(side_effect=httpx.NetworkError("boom"))

    mbps = await probe_bandwidth("10.0.0.1", client, api_port=52415)

    assert mbps is None
