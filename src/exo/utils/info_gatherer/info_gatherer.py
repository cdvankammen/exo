"""InfoGatherer: collects node information for topology and placement.

Gathers OS, network interfaces, backends, thunderbolt bridges, disk usage, and NVML/CUDA state into structured node info."""

import os
import re
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Self, cast

import anyio
from anyio import fail_after, open_process, to_thread
from anyio.streams.buffered import BufferedByteReceiveStream
from loguru import logger
from pydantic import ValidationError

from exo.shared.constants import EXO_CONFIG_FILE, EXO_DEFAULT_MODELS_DIR
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NetworkInterfaceInfo,
    ThunderboltBridgeStatus,
)
from exo.shared.types.thunderbolt import (
    ThunderboltConnection,
    ThunderboltConnectivity,
    ThunderboltIdentifier,
)
from exo.utils.channels import Sender
from exo.utils.pydantic_ext import TaggedModel
from exo.utils.task_group import TaskGroup

from .macmon import MacmonMetrics
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
    get_os_build_version,
    get_os_version,
)

IS_DARWIN = sys.platform == "darwin"


async def _get_thunderbolt_devices() -> set[str] | None:
    """Get Thunderbolt interface device names (e.g., en2, en3) from hardware ports.

    Returns None if the networksetup command fails.
    """
    result = await anyio.run_process(
        ["networksetup", "-listallhardwareports"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -listallhardwareports failed with code "
            f"{result.returncode}: {result.stderr.decode()}"
        )
        return None

    output = result.stdout.decode()
    thunderbolt_devices: set[str] = set()
    current_port: str | None = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            current_port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and current_port:
            device = line.split(":", 1)[1].strip()
            if "thunderbolt" in current_port.lower():
                thunderbolt_devices.add(device)
            current_port = None

    return thunderbolt_devices


async def _get_bridge_services() -> dict[str, str] | None:
    """Get mapping of bridge device -> service name from network service order.

    Returns None if the networksetup command fails.
    """
    result = await anyio.run_process(
        ["networksetup", "-listnetworkserviceorder"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -listnetworkserviceorder failed with code "
            f"{result.returncode}: {result.stderr.decode()}"
        )
        return None

    # Parse service order to find bridge devices and their service names
    # Format: "(1) Service Name\n(Hardware Port: ..., Device: bridge0)\n"
    service_order_output = result.stdout.decode()
    bridge_services: dict[str, str] = {}  # device -> service name
    current_service: str | None = None

    for line in service_order_output.splitlines():
        line = line.strip()
        # Match "(N) Service Name" or "(*) Service Name" (disabled)
        # but NOT "(Hardware Port: ...)" lines
        if (
            line
            and line.startswith("(")
            and ")" in line
            and not line.startswith("(Hardware Port:")
        ):
            paren_end = line.index(")")
            if paren_end + 2 <= len(line):
                current_service = line[paren_end + 2 :]
        # Match "(Hardware Port: ..., Device: bridgeX)"
        elif current_service and "Device: bridge" in line:
            # Extract device name from "..., Device: bridge0)"
            device_start = line.find("Device: ") + len("Device: ")
            device_end = line.find(")", device_start)
            if device_end > device_start:
                device = line[device_start:device_end]
                bridge_services[device] = current_service

    return bridge_services


async def _get_bridge_members(bridge_device: str) -> set[str]:
    """Get member interfaces of a bridge device via ifconfig."""
    result = await anyio.run_process(
        ["ifconfig", bridge_device],
        check=False,
    )
    if result.returncode != 0:
        logger.debug(f"ifconfig {bridge_device} failed with code {result.returncode}")
        return set()

    members: set[str] = set()
    ifconfig_output = result.stdout.decode()
    for line in ifconfig_output.splitlines():
        line = line.strip()
        if line.startswith("member:"):
            parts = line.split()
            if len(parts) > 1:
                members.add(parts[1])

    return members


async def _find_thunderbolt_bridge(
    bridge_services: dict[str, str], thunderbolt_devices: set[str]
) -> str | None:
    """Find the service name of a bridge containing Thunderbolt interfaces.

    Returns the service name if found, None otherwise.
    """
    for bridge_device, service_name in bridge_services.items():
        members = await _get_bridge_members(bridge_device)
        if members & thunderbolt_devices:  # intersection is non-empty
            return service_name
    return None


async def _is_service_enabled(service_name: str) -> bool | None:
    """Check if a network service is enabled.

    Returns True if enabled, False if disabled, None on error.
    """
    result = await anyio.run_process(
        ["networksetup", "-getnetworkserviceenabled", service_name],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -getnetworkserviceenabled '{service_name}' "
            f"failed with code {result.returncode}: {result.stderr.decode()}"
        )
        return None

    stdout = result.stdout.decode().strip().lower()
    return stdout == "enabled"


class StaticNodeInformation(TaggedModel):
    """Node information that should NEVER change, to be gathered once at startup"""

    model: str
    chip: str
    os_version: str
    os_build_version: str

    @classmethod
    async def gather(cls) -> Self:
        model, chip = await get_model_and_chip()
        return cls(
            model=model,
            chip=chip,
            os_version=get_os_version(),
            os_build_version=await get_os_build_version(),
        )


class NodeNetworkInterfaces(TaggedModel):
    ifaces: Sequence[NetworkInterfaceInfo]


class MacThunderboltIdentifiers(TaggedModel):
    idents: Sequence[ThunderboltIdentifier]


class MacThunderboltConnections(TaggedModel):
    conns: Sequence[ThunderboltConnection]


_IBV_DEVICE_LINE = re.compile(r"^\s*(\S+)\s+([0-9a-fA-F]{16})\s*$")

# ibv_devinfo prints one section per device with a per-port state line, e.g.
# "port 1: state: ACTIVE (4)" or "state ACTIVE". Treat any of these as a
# usable (link-up) port.
_IBV_ACTIVE_STATE_PATTERNS = (
    "state: ACTIVE",
    "state ACTIVE",
    "PORT_ACTIVE",
    "state ACTIVE (4)",
)


class RdmaCtlStatus(TaggedModel):
    enabled: bool
    has_verbs_device: bool = True

    @classmethod
    async def gather(cls) -> Self | None:
        if not IS_DARWIN or shutil.which("rdma_ctl") is None:
            return None
        try:
            with anyio.fail_after(5):
                proc = await anyio.run_process(["rdma_ctl", "status"], check=False)
        except (TimeoutError, OSError):
            return None
        if proc.returncode != 0:
            return None
        output = proc.stdout.decode("utf-8").lower().strip()
        if "enabled" not in output and "disabled" not in output:
            return None
        return cls(
            enabled="enabled" in output,
            has_verbs_device=await cls._gather_has_verbs_device(),
        )

    @staticmethod
    async def _gather_has_verbs_device() -> bool:
        """True if `ibv_devices` enumerates at least one RDMA verbs device
        AND `ibv_devinfo` reports an active port.

        `rdma_ctl status` can report "enabled" while no verbs device is exposed
        (Apple gating / unprovisioned Thunderbolt RDMA). jaccl then crashes with
        a NULL protection-domain dereference instead of failing cleanly
        (ml-explore/mlx#3777). A device can also be enumerated but its port not
        yet link-up — `ibv_devinfo` validates the port state, so placement can
        reject RDMA-backed instances until RDMA is actually usable.
        """
        if not IS_DARWIN or shutil.which("ibv_devices") is None:
            return False
        try:
            with anyio.fail_after(5):
                proc = await anyio.run_process(["ibv_devices"], check=False)
        except (TimeoutError, OSError):
            return False
        if proc.returncode != 0:
            return False
        output = proc.stdout.decode("utf-8", errors="replace")
        if not any(_IBV_DEVICE_LINE.match(line) for line in output.splitlines()):
            return False
        # Devices are enumerated — confirm at least one port is ACTIVE via
        # ibv_devinfo. Fail open if ibv_devinfo is missing or itself errors:
        # the ibv_devices enumeration is still authoritative for device
        # presence, and we don't want a validator regression to re-enable
        # placement on a device that never existed.
        if shutil.which("ibv_devinfo") is None:
            return True
        try:
            with anyio.fail_after(5):
                dev_proc = await anyio.run_process(["ibv_devinfo"], check=False)
        except (TimeoutError, OSError):
            return True
        if dev_proc.returncode != 0:
            return True
        dev_output = dev_proc.stdout.decode("utf-8", errors="replace")
        return any(pattern in dev_output for pattern in _IBV_ACTIVE_STATE_PATTERNS)


class ThunderboltBridgeInfo(TaggedModel):
    status: ThunderboltBridgeStatus

    @classmethod
    async def gather(cls) -> Self | None:
        """Check if a Thunderbolt Bridge network service is enabled on this node.

        Detection approach:
        1. Find all Thunderbolt interface devices (en2, en3, etc.) from hardware ports
        2. Find bridge devices from network service order (not hardware ports, as
           bridges may not appear there)
        3. Check each bridge's members via ifconfig
        4. If a bridge contains Thunderbolt interfaces, it's a Thunderbolt Bridge
        5. Check if that network service is enabled
        """
        if not IS_DARWIN:
            return None

        def _no_bridge_status() -> Self:
            return cls(
                status=ThunderboltBridgeStatus(
                    enabled=False, exists=False, service_name=None
                )
            )

        try:
            tb_devices = await _get_thunderbolt_devices()
            if tb_devices is None:
                return _no_bridge_status()

            bridge_services = await _get_bridge_services()
            if not bridge_services:
                return _no_bridge_status()

            tb_service_name = await _find_thunderbolt_bridge(
                bridge_services, tb_devices
            )
            if not tb_service_name:
                return _no_bridge_status()

            enabled = await _is_service_enabled(tb_service_name)
            if enabled is None:
                return cls(
                    status=ThunderboltBridgeStatus(
                        enabled=False, exists=True, service_name=tb_service_name
                    )
                )

            return cls(
                status=ThunderboltBridgeStatus(
                    enabled=enabled,
                    exists=True,
                    service_name=tb_service_name,
                )
            )
        except Exception as e:
            logger.opt(exception=e).warning("Failed to gather Thunderbolt Bridge info")
            return None


class NodeConfig(TaggedModel):
    """Node configuration from EXO_CONFIG_FILE, reloaded from the file only at startup. Other changes should come in through the API and propagate from there"""

    @classmethod
    async def gather(cls) -> Self | None:
        cfg_file = anyio.Path(EXO_CONFIG_FILE)
        await cfg_file.parent.mkdir(parents=True, exist_ok=True)
        await cfg_file.touch(exist_ok=True)
        async with await cfg_file.open("rb") as f:
            try:
                contents = (await f.read()).decode("utf-8")
                data = tomllib.loads(contents)
                return cls.model_validate(data)
            except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError):
                logger.warning("Invalid config file, skipping...")
                return None


class MiscData(TaggedModel):
    """Node information that may slowly change that doesn't fall into the other categories"""

    friendly_name: str

    @classmethod
    async def gather(cls) -> Self:
        return cls(friendly_name=await get_friendly_name())


class NodeApiInfo(TaggedModel):
    """Where this node's HTTP API listens, so cluster-wide views (logs/errors)
    can reach every node. Populated once at startup from the node's launch args."""

    api_host: str
    api_port: int

    @classmethod
    async def gather(cls) -> Self | None:
        from exo.shared.constants import EXO_API_ADVERTISE_HOST, EXO_API_HOST

        api_port = int(os.getenv("EXO_API_PORT", "52415"))
        return cls(api_host=await resolve_advertise_host(EXO_API_HOST, EXO_API_ADVERTISE_HOST), api_port=api_port)


async def resolve_advertise_host(bind_host: str, advertise_override: str | None) -> str:
    """Pick the address peers should use to reach this node's HTTP API.

    ``advertise_override`` (EXO_API_ADVERTISE_HOST) wins when set. Otherwise
    wildcard bind hosts (0.0.0.0 / ::) and loopback binds (127.0.0.1 / ::1)
    both fall back to the node's primary non-loopback IP — peers can neither
    dial 0.0.0.0 nor reach another host's 127.0.0.1. A concrete bind host is
    used verbatim.
    """
    if advertise_override:
        return advertise_override
    if bind_host in ("0.0.0.0", "::", "127.0.0.1", "::1", ""):
        return await _primary_non_loopback_ip()
    return bind_host


async def _primary_non_loopback_ip() -> str:
    """Best-effort primary non-loopback IPv4 of this host (LAN/Tailscale)."""
    import socket

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip: str = str(info[4][0])
            if not ip.startswith("127."):
                return ip
    except OSError:
        pass
    # Fallback: enumerate interfaces via psutil (no DNS dependency).
    try:
        import psutil

        for iface, addrs in psutil.net_if_addrs().items():
            if iface == "lo" or iface.startswith("lo"):
                continue
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    candidate = str(addr.address)
                    if not candidate.startswith("127."):
                        return candidate
    except Exception:
        pass
    return "127.0.0.1"


class NodeDiskUsage(TaggedModel):
    """Disk space information for the models directory."""

    disk_usage: DiskUsage

    @classmethod
    async def gather(cls) -> Self:
        return cls(
            disk_usage=await to_thread.run_sync(
                DiskUsage.from_path, EXO_DEFAULT_MODELS_DIR
            )
        )


async def _gather_iface_map() -> dict[str, str] | None:
    proc = await anyio.run_process(
        ["networksetup", "-listallhardwareports"], check=False
    )
    if proc.returncode != 0:
        return None

    ports: dict[str, str] = {}
    port = ""
    for line in proc.stdout.decode("utf-8").split("\n"):
        if line.startswith("Hardware Port:"):
            port = line.split(": ")[1]
        elif line.startswith("Device:"):
            ports[port] = line.split(": ")[1]
            port = ""
    if "" in ports:
        del ports[""]
    return ports


def _has_nvml_cuda() -> bool:
    try:
        import pynvml as nvml  # pyright: ignore[reportMissingModuleSource]
    except ImportError:
        return False
    try:
        nvml.nvmlInit()
        try:
            return nvml.nvmlDeviceGetCount() > 0
        finally:
            nvml.nvmlShutdown()
    except Exception:
        return False


class NodeBackends(TaggedModel):
    backends: list[Backend]

    @classmethod
    async def gather(cls) -> Self:
        backends: list[Backend] = [Backend.MlxCpu]
        if IS_DARWIN:
            backends.append(Backend.MlxMetal)
        if await to_thread.run_sync(_has_nvml_cuda) or _mlx_uses_cuda_gpu():
            backends.append(Backend.MlxCuda)
            backends.append(Backend.Vllm)
        return cls(backends=backends)


GatheredInfo = (
    MacmonMetrics
    | MemoryUsage
    | NodeNetworkInterfaces
    | MacThunderboltIdentifiers
    | MacThunderboltConnections
    | RdmaCtlStatus
    | ThunderboltBridgeInfo
    | NodeConfig
    | MiscData
    | NodeApiInfo
    | StaticNodeInformation
    | NodeDiskUsage
    | NodeBackends
)


def _mlx_uses_cuda_gpu() -> bool:
    """True when MLX's default device is a (CUDA) GPU on a non-Darwin host.

    Used to decide whether to report GPU VRAM instead of system RAM as the
    node's memory. On Apple Silicon the GPU path is handled by macmon, and on a
    Linux CPU (mlx-cpu) build the default device is the CPU, so both correctly
    fall through to psutil system RAM.
    """
    if IS_DARWIN:
        return False
    try:
        import mlx.core as mx

        return "gpu" in str(mx.default_device()).lower()
    except Exception:
        return False


@dataclass
class InfoGatherer:
    info_sender: Sender[GatheredInfo]
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _psutil_enabled: bool = field(init=False, default=False)

    async def _can_read_macmon_metrics(self, macmon_path: str) -> bool:
        try:
            with fail_after(5):
                proc = await anyio.run_process(
                    [macmon_path, "pipe", "--samples", "1", "--interval", "100"],
                    check=False,
                )
        except Exception as e:
            logger.opt(exception=e).warning(
                f"Failed to validate macmon at {macmon_path}"
            )
            return False

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            logger.warning(
                f"macmon preflight failed with return code {proc.returncode}: "
                f"{stderr or 'no stderr'}"
            )
            return False

        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        if not stdout:
            logger.warning("macmon preflight returned no metrics")
            return False

        try:
            MacmonMetrics.from_raw_json(stdout.splitlines()[0])
        except ValidationError as e:
            logger.opt(exception=e).warning(
                "macmon preflight returned unexpected metrics JSON"
            )
            return False

        return True

    async def run(self):
        async with self._tg as tg:
            if IS_DARWIN:
                tg.start_soon(self._monitor_macmon, 1)
                tg.start_soon(self._monitor_system_profiler_thunderbolt_data, 5)
                tg.start_soon(self._monitor_thunderbolt_bridge_status, 10)
                tg.start_soon(self._monitor_rdma_ctl_status, 10)
            if not IS_DARWIN:
                tg.start_soon(self._monitor_memory_usage, 1)
            tg.start_soon(self._watch_system_info, 10)
            tg.start_soon(self._monitor_misc, 60)
            tg.start_soon(self._monitor_static_info, 60)
            tg.start_soon(self._monitor_disk_usage, 30)
            tg.start_soon(self._monitor_node_backends, 60)
            tg.start_soon(self._monitor_node_config, 60)
            tg.start_soon(self._monitor_node_api_info, 60)

            nc = await NodeConfig.gather()
            if nc is not None:
                await self.info_sender.send(nc)

            # Advertise this node's API endpoint so cluster-wide views
            # (logs, errors) can reach it over HTTP.
            api_info = await NodeApiInfo.gather()
            if api_info is not None:
                await self.info_sender.send(api_info)

            try:
                await self.info_sender.send(await NodeBackends.gather())
            except Exception as e:
                logger.opt(exception=e).warning(
                    "Error gathering node backends at startup"
                )

    def shutdown(self):
        self._tg.cancel_tasks()

    async def _monitor_static_info(self, static_info_poll_interval: float):
        while True:
            try:
                with fail_after(30):
                    await self.info_sender.send(await StaticNodeInformation.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering static node info")
            await anyio.sleep(static_info_poll_interval)

    async def _monitor_node_backends(self, backends_poll_interval: float):
        """Periodically re-advertise this node's backends.

        NodeBackends is only gathered once at startup; if that first message
        is lost (e.g. the router is still connecting when the gatherer
        starts), the node advertises no backends and placement rejects every
        cycle containing it — permanently, until the app restarts. Re-sending
        on an interval makes the advertisement self-healing.
        """
        while True:
            try:
                with fail_after(30):
                    await self.info_sender.send(await NodeBackends.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering node backends")
            await anyio.sleep(backends_poll_interval)

    async def _monitor_node_config(self, config_poll_interval: float):
        """Periodically re-advertise this node's NodeConfig (identity).

        NodeConfig is only gathered once at startup. If that first message is
        lost, the node can join elections and be 'discovered' by peers yet
        never appear in any node's topology (seen live 08-07: Linux
        9519a896). Re-sending on an interval makes the identity self-healing.
        """
        while True:
            try:
                with fail_after(30):
                    nc = await NodeConfig.gather()
                    if nc is not None:
                        await self.info_sender.send(nc)
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering node config")
            await anyio.sleep(config_poll_interval)

    async def _monitor_node_api_info(self, api_info_poll_interval: float):
        """Periodically re-advertise this node's NodeApiInfo (API host/port).

        NodeApiInfo is only gathered once at startup. If that first message is
        lost (router still connecting) or the node restarts with a different
        API port, peers keep probing the node at the wrong port and the
        topology edge-deletion loop (worker/main.py ``_poll_connection_updates``)
        removes any edge whose sink port doesn't match the advertised
        ``expected_port`` — leaving the node with zero edges until every peer
        restarts. Seen live 09-01: tryingexo Linux containers advertised
        ``apiPort: 0`` and formed no outgoing edges. Re-sending on an interval
        makes the API-port announcement self-healing.
        """
        while True:
            try:
                with fail_after(30):
                    api_info = await NodeApiInfo.gather()
                    if api_info is not None:
                        await self.info_sender.send(api_info)
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering node api info")
            await anyio.sleep(api_info_poll_interval)

    async def _monitor_misc(self, misc_poll_interval: float):
        while True:
            try:
                with fail_after(10):
                    await self.info_sender.send(await MiscData.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering misc data")
            await anyio.sleep(misc_poll_interval)

    async def _monitor_system_profiler_thunderbolt_data(
        self, system_profiler_interval: float
    ):
        while True:
            try:
                with fail_after(30):
                    iface_map = await _gather_iface_map()
                    if iface_map is None:
                        raise ValueError("Failed to gather interface map")

                    data = await ThunderboltConnectivity.gather()
                    assert data is not None

                    idents = [
                        it for i in data if (it := i.ident(iface_map)) is not None
                    ]
                    await self.info_sender.send(
                        MacThunderboltIdentifiers(idents=idents)
                    )

                    conns = [it for i in data if (it := i.conn()) is not None]
                    await self.info_sender.send(MacThunderboltConnections(conns=conns))
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering Thunderbolt data")
            await anyio.sleep(system_profiler_interval)

    async def _monitor_memory_usage(self, memory_poll_rate: float):
        if self._psutil_enabled:
            return
        self._psutil_enabled = True
        override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
        override_memory: int | None = (
            Memory.from_mb(int(override_memory_env)).in_bytes
            if override_memory_env
            else None
        )
        report_vram = _mlx_uses_cuda_gpu()
        if report_vram:
            logger.info("CUDA MLX backend detected; reporting GPU VRAM as node memory")
        while True:
            try:
                usage: MemoryUsage | None = None
                if report_vram:
                    # Bound the CUDA VRAM query: a hung nvidia-smi (GPU driver in
                    # D-state) must not stall this monitor indefinitely. Without a
                    # deadline, a slow subprocess inside from_cuda() blocks the
                    # loop forever and can hold up graceful shutdown.
                    with fail_after(10):
                        usage = MemoryUsage.from_cuda(override_memory=override_memory)
                if usage is None:
                    usage = MemoryUsage.from_system(override_memory=override_memory)
                await self.info_sender.send(usage)
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering memory usage")
            await anyio.sleep(memory_poll_rate)

    async def _watch_system_info(self, interface_watcher_interval: float):
        while True:
            try:
                with fail_after(10):
                    nics = await get_network_interfaces()
                    await self.info_sender.send(NodeNetworkInterfaces(ifaces=nics))
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering network interfaces")
            await anyio.sleep(interface_watcher_interval)

    async def _monitor_thunderbolt_bridge_status(
        self, thunderbolt_bridge_poll_interval: float
    ):
        while True:
            try:
                with fail_after(30):
                    curr = await ThunderboltBridgeInfo.gather()
                    if curr is not None:
                        await self.info_sender.send(curr)
            except Exception as e:
                logger.opt(exception=e).warning(
                    "Error gathering Thunderbolt Bridge status"
                )
            await anyio.sleep(thunderbolt_bridge_poll_interval)

    async def _monitor_rdma_ctl_status(self, rdma_ctl_poll_interval: float):
        while True:
            try:
                curr = await RdmaCtlStatus.gather()
                if curr is not None:
                    await self.info_sender.send(curr)
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering RDMA ctl status")
            await anyio.sleep(rdma_ctl_poll_interval)

    async def _monitor_disk_usage(self, disk_poll_interval: float):
        while True:
            try:
                with fail_after(5):
                    await self.info_sender.send(await NodeDiskUsage.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering disk usage")
            await anyio.sleep(disk_poll_interval)

    async def _monitor_macmon(self, macmon_interval: float):
        # Single-active-monitor semantics: only ONE memory event source may
        # run at a time (macmon OR psutil fallback). A fallback is armed
        # before the retry loop is entered; the loop then RETURNS instead of
        # re-arming or continuing, so a macmon crash cannot leave both a
        # retrying macmon loop and a psutil monitor publishing conflicting
        # MemoryUsage/MacmonMetrics events (the pre-existing _psutil_enabled
        # flag alone never stopped the outer while-loop from re-running).
        if (
            macmon_path := os.getenv("EXO_MACMON_PATH") or shutil.which("macmon")
        ) is None:
            logger.warning(
                "macmon not found, falling back to psutil for memory monitoring"
            )
            if not self._psutil_enabled:
                self._tg.start_soon(self._monitor_memory_usage, 1)
            return
        if not await self._can_read_macmon_metrics(macmon_path):
            logger.warning(
                f"macmon at {macmon_path} is unusable, falling back to psutil memory monitoring"
            )
            if not self._psutil_enabled:
                self._tg.start_soon(self._monitor_memory_usage, 1)
            return
        # macmon pipe --interval [interval in ms]
        # Timeout: if macmon produces no output for this many seconds, restart it.
        # macmon writes every macmon_interval seconds, so 10x that is generous.
        read_timeout = max(macmon_interval * 10, 30)
        while True:
            try:
                async with await open_process(
                    [
                        macmon_path,
                        "pipe",
                        "--interval",
                        str(macmon_interval * 1000),
                    ]
                ) as p:
                    if not p.stdout:
                        logger.critical("MacMon closed stdout")
                        return
                    stream = BufferedByteReceiveStream(p.stdout)
                    while True:
                        with fail_after(read_timeout):
                            data = await stream.receive_until(
                                delimiter=b"\n", max_bytes=8 * 1024
                            )
                            text = data.decode("utf-8", errors="replace").strip()
                            if not text:
                                # macmon can emit an occasional empty line;
                                # skip it rather than failing JSON parse.
                                continue
                            # macmon output is line-delimited: one JSON
                            # object per line (preflight validates line 0).
                            # Parse per-line so a stray multi-line blob or
                            # trailing whitespace never corrupts the stream.
                            metrics = MacmonMetrics.from_raw_json(text)
                        await self.info_sender.send(metrics)
            except TimeoutError:
                logger.warning(
                    f"MacMon produced no output for {read_timeout}s, restarting"
                )
                # Hand over to the psutil fallback; do NOT keep retrying
                # macmon alongside it (single-active-monitor semantics).
                if not self._psutil_enabled:
                    self._tg.start_soon(self._monitor_memory_usage, 1)
                return
            except CalledProcessError as e:
                stderr_msg = "no stderr"
                stderr_output = cast(bytes | str | None, e.stderr)
                if stderr_output is not None:
                    stderr_msg = (
                        stderr_output.decode()
                        if isinstance(stderr_output, bytes)
                        else str(stderr_output)
                    )
                logger.warning(
                    f"MacMon failed with return code {e.returncode}: {stderr_msg}"
                )
                if not self._psutil_enabled:
                    self._tg.start_soon(self._monitor_memory_usage, 1)
                return
            except ProcessLookupError:
                # usually throws by the process' context manager on exit
                # when we ctrl+c, hence usually should be ignored;
                # if anything else throws it, we explicitly don't care:
                # process is dead anyways ;)
                logger.warning(
                    "Macmon process not found - shutting down macmon monitor"
                )
                return
            except Exception as e:
                logger.opt(exception=e).warning("Error in macmon monitor")
                if not self._psutil_enabled:
                    self._tg.start_soon(self._monitor_memory_usage, 1)
                return
