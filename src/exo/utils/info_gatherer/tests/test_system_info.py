"""Branch-coverage tests for system_info helpers.

Most failure paths are platform/process dependent, so we exercise them with
monkeypatched subprocess results and platform flags. Covers the Linux
interface classification, CUDA GPU name lookup, macOS system_profiler parse,
and the networksetup parsing branches.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.utils.info_gatherer import system_info as si


# ---- get_os_version / get_os_build_version --------------------------------


def test_get_os_version_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with patch("platform.mac_ver", return_value=("15.3", ("x",), "arm64")):
        assert si.get_os_version() == "15.3"


def test_get_os_version_darwin_empty(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with patch("platform.mac_ver", return_value=("", (), "")):
        assert si.get_os_version() == "Unknown"


def test_get_os_version_linux(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with patch("platform.system", return_value="Linux"):
        assert si.get_os_version() == "Linux"


async def test_get_os_build_version_non_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert await si.get_os_build_version() == "Unknown"


async def test_get_os_build_version_darwin_ok(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    process = MagicMock()
    process.stdout = b"24D5055b\n"
    with patch("exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)):
        assert await si.get_os_build_version() == "24D5055b"


async def test_get_os_build_version_darwin_calledprocesserror(monkeypatch) -> None:
    from subprocess import CalledProcessError

    monkeypatch.setattr(sys, "platform", "darwin")
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=CalledProcessError(1, "sw_vers")),
    ):
        assert await si.get_os_build_version() == "Unknown"


async def test_get_os_build_version_darwin_empty_stdout(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    process = MagicMock()
    process.stdout = b"   \n"
    with patch("exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)):
        assert await si.get_os_build_version() == "Unknown"


# ---- get_friendly_name -----------------------------------------------------


async def test_get_friendly_name_non_darwin_returns_hostname(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with patch("socket.gethostname", return_value="myhost"):
        assert await si.get_friendly_name() == "myhost"


async def test_get_friendly_name_darwin_ok(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    process = MagicMock()
    process.stdout = b"Chris's MacBook Pro\n"
    with patch("socket.gethostname", return_value="chris-mbp"), patch(
        "exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)
    ):
        assert await si.get_friendly_name() == "Chris's MacBook Pro"


async def test_get_friendly_name_darwin_error_falls_back(monkeypatch) -> None:
    from subprocess import CalledProcessError

    monkeypatch.setattr(sys, "platform", "darwin")
    with patch("socket.gethostname", return_value="chris-mbp"), patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=CalledProcessError(1, "scutil")),
    ):
        assert await si.get_friendly_name() == "chris-mbp"


# ---- interface classification ---------------------------------------------


async def test_get_network_interfaces_linux_classification(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    # Build a fake /sys/class/net tree under tmp_path
    sysfs = tmp_path / "net"
    (sysfs / "eth0" / "device").mkdir(parents=True)
    (sysfs / "eth0" / "speed").write_text("1000")
    (sysfs / "wlan0" / "wireless").mkdir(parents=True)
    (sysfs / "wlan0" / "speed").write_text("300")
    (sysfs / "enp0s3f0np0").mkdir(parents=True)
    (sysfs / "enp0s3f0np0" / "speed").write_text("-1")
    (sysfs / "tb0" / "device").mkdir(parents=True)
    (sysfs / "br-int").mkdir(parents=True)

    real_path = Path

    def fake_path(p: str) -> Path:
        if p == "/sys/class/net":
            return sysfs
        return real_path(p)

    monkeypatch.setattr(si, "Path", fake_path)

    def net_if_addrs():
        return {
            "eth0": [MagicMock(family=2, address="10.0.0.1")],
            "wlan0": [MagicMock(family=2, address="10.0.0.2")],
            "enp0s3f0np0": [MagicMock(family=2, address="10.0.0.3")],
            "lo": [MagicMock(family=2, address="127.0.0.1")],
            "tb0": [MagicMock(family=2, address="10.0.0.4")],
            "br-int": [MagicMock(family=2, address="10.0.0.5")],
            "docker0": [MagicMock(family=2, address="172.17.0.1")],
            "sit0": [MagicMock(family=2, address="10.0.0.6")],
        }

    from types import SimpleNamespace

    def net_io(pernic=True):
        return {
            k: SimpleNamespace(bytes_recv=100, bytes_sent=200)
            for k in ("eth0", "wlan0", "enp0s3f0np0", "tb0", "sit0")
        }

    monkeypatch.setattr("psutil.net_if_addrs", net_if_addrs)
    monkeypatch.setattr("psutil.net_io_counters", net_io)

    result = await si.get_network_interfaces()

    by_name = {i.name: i for i in result}
    assert by_name["eth0"].interface_type == "ethernet"
    # tb0 has a device symlink and no wireless dir -> classified ethernet by
    # the sysfs classifier (the name-based tb->thunderbolt rule lives in
    # _linux_interface_types, which get_network_interfaces overrides on Linux)
    assert by_name["tb0"].interface_type == "ethernet"
    assert by_name["sit0"].interface_type == "unknown"
    # lo / docker0 / br-int are skipped by _linux_interface_types and have no
    # device symlink -> unknown if they appeared; loopback has no counters
    assert by_name["eth0"].link_speed_megabits == 1000
    assert by_name["wlan0"].interface_type == "wifi"
    assert by_name["wlan0"].link_speed_megabits is None  # non-positive
    assert by_name["enp0s3f0np0"].interface_type == "unknown"
    assert by_name["enp0s3f0np0"].link_speed_megabits is None  # negative


def test_classify_linux_interface_wifi(tmp_path) -> None:
    (tmp_path / "wireless").mkdir()
    assert si._classify_linux_interface(str(tmp_path)) == ("wifi", None)


def test_classify_linux_interface_speed_read_error(tmp_path) -> None:
    # missing speed file -> OSError -> None
    assert si._classify_linux_interface(str(tmp_path)) == ("unknown", None)


def test_classify_linux_interface_invalid_speed(tmp_path) -> None:
    (tmp_path / "speed").write_text("fast")
    assert si._classify_linux_interface(str(tmp_path)) == ("unknown", None)


def test_classify_linux_interface_device_symlink(tmp_path) -> None:
    (tmp_path / "speed").write_text("100")
    (tmp_path / "device").mkdir()
    assert si._classify_linux_interface(str(tmp_path)) == ("ethernet", 100)


def test_classify_linux_interface_negative_speed(tmp_path) -> None:
    (tmp_path / "speed").write_text("-1")
    (tmp_path / "device").mkdir()
    assert si._classify_linux_interface(str(tmp_path)) == ("ethernet", None)


# ---- networksetup parsing (macOS) -----------------------------------------


NETWORKSETUP_OUT = """Hardware Port: Wi-Fi
Device: en0

Hardware Port: Ethernet
Device: en5

Hardware Port: Thunderbolt Bridge
Device: bridge0

Hardware Port: USB 10/100/1000 LAN
Device: en6

Hardware Port: Other Port
Device: enx1234
""".encode()


async def test_networksetup_parse_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    process = MagicMock()
    process.stdout = NETWORKSETUP_OUT
    with patch("exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)):
        types = await si._get_interface_types_from_networksetup()
    assert types["en0"] == "wifi"  # en0 excluded from maybe_ethernet demotion
    assert types["en5"] == "maybe_ethernet"  # en-prefixed non-en0/en1 demoted
    assert types["bridge0"] == "thunderbolt"
    assert types["en6"] == "maybe_ethernet"  # en-prefixed demoted even with LAN port
    # en-prefixed non-en0/en1 -> maybe_ethernet
    assert types["enx1234"] == "maybe_ethernet"


async def test_networksetup_parse_failure_returns_empty(monkeypatch) -> None:
    from subprocess import CalledProcessError

    monkeypatch.setattr(sys, "platform", "darwin")
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=CalledProcessError(1, "networksetup")),
    ):
        assert await si._get_interface_types_from_networksetup() == {}


async def test_networksetup_skipped_on_linux(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with patch(
        "exo.utils.info_gatherer.system_info._linux_interface_types",
        return_value={"eth0": "ethernet"},
    ):
        assert await si._get_interface_types_from_networksetup() == {
            "eth0": "ethernet"
        }


# ---- CUDA GPU name / model+chip -------------------------------------------


async def test_get_cuda_gpu_name_ok(monkeypatch) -> None:
    process = MagicMock()
    process.returncode = 0
    process.stdout = b"NVIDIA GeForce RTX 3090\n"
    with patch("exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)):
        assert await si._get_cuda_gpu_name() == "NVIDIA GeForce RTX 3090"


async def test_get_cuda_gpu_name_nonzero_returncode(monkeypatch) -> None:
    process = MagicMock()
    process.returncode = 1
    process.stdout = b""
    with patch("exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)):
        assert await si._get_cuda_gpu_name() is None


async def test_get_cuda_gpu_name_empty_output(monkeypatch) -> None:
    process = MagicMock()
    process.returncode = 0
    process.stdout = b""
    with patch("exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)):
        assert await si._get_cuda_gpu_name() is None


async def test_get_cuda_gpu_name_oserror(monkeypatch) -> None:
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=OSError("no nvidia-smi")),
    ):
        assert await si._get_cuda_gpu_name() is None


async def test_get_cuda_gpu_name_timeout_returns_none(monkeypatch) -> None:
    """A hung nvidia-smi (driver in D-state) must not stall the query forever.

    fail_after(5) raises TimeoutError; treat it as "no GPU name" instead of
    letting it propagate and stall static-node-info gathering indefinitely.
    """

    async def _hang(*_args: object, **_kwargs: object) -> object:
        import anyio

        with anyio.fail_after(0.05):
            await anyio.sleep(10)
        raise AssertionError("unreachable")

    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=_hang),
    ):
        # The patch drives fail_after's timeout down to ~50ms; the guard
        # converts the timeout into None rather than re-raising.
        result = await si._get_cuda_gpu_name()
        assert result is None


async def test_get_model_and_chip_linux_with_gpu(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with patch(
        "exo.utils.info_gatherer.system_info._get_cuda_gpu_name",
        AsyncMock(return_value="NVIDIA RTX A6000"),
    ):
        model, chip = await si.get_model_and_chip()
    assert model == "Unknown Model"
    assert chip == "NVIDIA RTX A6000"


async def test_get_model_and_chip_linux_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with patch(
        "exo.utils.info_gatherer.system_info._get_cuda_gpu_name",
        AsyncMock(return_value=None),
    ):
        model, chip = await si.get_model_and_chip()
    assert (model, chip) == ("Unknown Model", "Unknown Chip")


async def test_get_model_and_chip_darwin_ok(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    process = MagicMock()
    process.stdout = b'Model Name: Mac mini\nChip: Apple M4 Pro\n'
    with patch("exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)):
        model, chip = await si.get_model_and_chip()
    assert model == "Mac mini"
    assert chip == "Apple M4 Pro"


async def test_get_model_and_chip_darwin_calledprocesserror(monkeypatch) -> None:
    from subprocess import CalledProcessError

    monkeypatch.setattr(sys, "platform", "darwin")
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=CalledProcessError(1, "system_profiler")),
    ):
        assert await si.get_model_and_chip() == ("Unknown Model", "Unknown Chip")


async def test_get_model_and_chip_darwin_missing_lines(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    process = MagicMock()
    process.stdout = b"Memory: 32 GB\n"
    with patch("exo.utils.info_gatherer.system_info.run_process", AsyncMock(return_value=process)):
        assert await si.get_model_and_chip() == ("Unknown Model", "Unknown Chip")


# ---- get_network_interfaces address filtering ------------------------------

async def test_get_network_interfaces_filters_non_inet_families(monkeypatch) -> None:
    # Only AF_INET/AF_INET6 addresses are kept
    monkeypatch.setattr(sys, "platform", "darwin")
    lo = MagicMock()
    lo.family = 17  # AF_PACKET-ish: not inet
    eth = MagicMock()
    eth.family = 2  # AF_INET
    eth.address = "10.0.0.5"

    def net_if_addrs():
        return {"lo0": [lo], "en1": [eth]}

    def net_io(pernic=True):
        return {}

    monkeypatch.setattr("psutil.net_if_addrs", net_if_addrs)
    monkeypatch.setattr("psutil.net_io_counters", net_io)
    with patch(
        "exo.utils.info_gatherer.system_info._get_interface_types_from_networksetup",
        AsyncMock(return_value={"lo0": "unknown", "en1": "wifi"}),
    ):
        result = await si.get_network_interfaces()

    assert len(result) == 1
    assert result[0].name == "en1"
    assert result[0].ip_address == "10.0.0.5"
    assert result[0].interface_type == "wifi"


async def test_get_network_interfaces_missing_counters(monkeypatch) -> None:
    # counters absent for iface -> rx/tx None
    monkeypatch.setattr(sys, "platform", "darwin")
    eth = MagicMock()
    eth.family = 2
    eth.address = "10.0.0.5"

    def net_if_addrs():
        return {"en1": [eth]}

    def net_io(pernic=True):
        return {}

    monkeypatch.setattr("psutil.net_if_addrs", net_if_addrs)
    monkeypatch.setattr("psutil.net_io_counters", net_io)
    with patch(
        "exo.utils.info_gatherer.system_info._get_interface_types_from_networksetup",
        AsyncMock(return_value={}),
    ):
        result = await si.get_network_interfaces()
    assert result[0].rx_bytes is None
    assert result[0].tx_bytes is None
    assert result[0].interface_type == "unknown"  # default when not in map


def test_linux_interface_types_classifies_by_name(monkeypatch) -> None:
    def net_if_addrs():
        return {
            "lo": [MagicMock()],
            "docker0": [MagicMock()],
            "br-int": [MagicMock()],
            "wlan0": [MagicMock()],
            "enp0s3f0np0": [MagicMock()],
            "en5": [MagicMock()],
            "tb0": [MagicMock()],
            "zzz": [MagicMock()],
        }

    monkeypatch.setattr("psutil.net_if_addrs", net_if_addrs)
    types = si._linux_interface_types()
    # lo / docker / br- are skipped entirely
    assert "lo" not in types
    assert "docker0" not in types
    assert "br-int" not in types
    assert types["wlan0"] == "wifi"
    assert types["enp0s3f0np0"] == "ethernet"
    assert types["en5"] == "maybe_ethernet"
    assert types["tb0"] == "thunderbolt"
    assert types["zzz"] == "unknown"