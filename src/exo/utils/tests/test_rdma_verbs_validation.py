"""Tests for RdmaCtlStatus verbs-device validation (ibv_devices + ibv_devinfo)."""

from unittest.mock import AsyncMock, patch

import pytest

from exo.utils.info_gatherer.info_gatherer import RdmaCtlStatus


class _Proc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout.encode()
        self.returncode = returncode


def _run_process_side_effect(outputs: dict[tuple[str, ...], tuple[str, int]]):
    """Side effect factory mapping argv tuples to (stdout, returncode)."""

    async def _fake_run_process(
        command: list[str], check: bool = False
    ) -> _Proc:
        key = tuple(command)
        if key in outputs:
            stdout, rc = outputs[key]
            return _Proc(stdout, rc)
        return _Proc("", 1)

    return _fake_run_process


@pytest.fixture
def mock_run_process(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr(
        "exo.utils.info_gatherer.info_gatherer.anyio.run_process", mock
    )
    return mock


def _ibv_devices_output(devices: list[str]) -> str:
    lines = ["    device                 node GUID", "    ------              ----------------"]
    for i, dev in enumerate(devices):
        lines.append(f"    {dev}                 {i:016x}")
    return "\n".join(lines) + "\n"


def _ibv_devinfo_active_output() -> str:
    return (
        "hca_id: rdma_en1\n"
        "\ttransport: InfiniBand (0)\n"
        "\tport: 1\n"
        "\tstate: ACTIVE (4)\n"
        "\tphysical_state: LINK_UP (5)\n"
    )


def _ibv_devinfo_down_output() -> str:
    return (
        "hca_id: rdma_en1\n"
        "\tport: 1\n"
        "\tstate: DOWN (1)\n"
        "\tphysical_state: DISABLED (1)\n"
    )


async def test_gather_false_when_ibv_devices_empty(
    mock_run_process: AsyncMock,
) -> None:
    mock_run_process.side_effect = _run_process_side_effect(
        {
            ("rdma_ctl", "status"): ("enabled\n", 0),
            ("ibv_devices",): (_ibv_devices_output([]), 0),
        },
    )

    status = await RdmaCtlStatus.gather()

    assert status is not None
    assert status.enabled is True
    assert status.has_verbs_device is False


async def test_gather_true_when_device_and_active_port(
    mock_run_process: AsyncMock,
) -> None:
    mock_run_process.side_effect = _run_process_side_effect(
        {
            ("rdma_ctl", "status"): ("enabled\n", 0),
            ("ibv_devices",): (_ibv_devices_output(["rdma_en1"]), 0),
            ("ibv_devinfo",): (_ibv_devinfo_active_output(), 0),
        },
    )

    status = await RdmaCtlStatus.gather()

    assert status is not None
    assert status.has_verbs_device is True


async def test_gather_false_when_device_but_port_down(
    mock_run_process: AsyncMock,
) -> None:
    """Device enumerated but port DOWN must NOT qualify (TODO #28)."""
    mock_run_process.side_effect = _run_process_side_effect(
        {
            ("rdma_ctl", "status"): ("enabled\n", 0),
            ("ibv_devices",): (_ibv_devices_output(["rdma_en1"]), 0),
            ("ibv_devinfo",): (_ibv_devinfo_down_output(), 0),
        },
    )

    status = await RdmaCtlStatus.gather()

    assert status is not None
    assert status.has_verbs_device is False


async def test_gather_fail_open_when_ibv_devinfo_missing(
    mock_run_process: AsyncMock,
) -> None:
    """ibv_devinfo unavailable -> fall back to ibv_devices enumeration only."""
    mock_run_process.side_effect = _run_process_side_effect(
        {
            ("rdma_ctl", "status"): ("enabled\n", 0),
            ("ibv_devices",): (_ibv_devices_output(["rdma_en1"]), 0),
        },
    )

    def _which(name: str) -> str | None:
        if name == "rdma_ctl":
            return "/usr/bin/rdma_ctl"
        if name == "ibv_devices":
            return "/usr/bin/ibv_devices"
        return None

    with patch(
        "exo.utils.info_gatherer.info_gatherer.shutil.which",
        side_effect=_which,
    ) as mock_which:
        assert mock_which is not None
        status = await RdmaCtlStatus.gather()

    assert status is not None
    assert status.has_verbs_device is True


async def test_gather_fail_open_when_ibv_devinfo_errors(
    mock_run_process: AsyncMock,
) -> None:
    mock_run_process.side_effect = _run_process_side_effect(
        {
            ("rdma_ctl", "status"): ("enabled\n", 0),
            ("ibv_devices",): (_ibv_devices_output(["rdma_en1"]), 0),
            ("ibv_devinfo",): ("ibv_devinfo: device not found\n", 1),
        },
    )

    status = await RdmaCtlStatus.gather()

    assert status is not None
    assert status.has_verbs_device is True


async def test_gather_disabled_rdma_ctl(
    mock_run_process: AsyncMock,
) -> None:
    mock_run_process.side_effect = _run_process_side_effect(
        {
            ("rdma_ctl", "status"): ("disabled\n", 0),
        },
    )

    status = await RdmaCtlStatus.gather()

    assert status is not None
    assert status.enabled is False
