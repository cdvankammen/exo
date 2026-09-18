"""macmon metrics parsing.

Parses raw macmon output into structured memory/performance metrics (:class:`MacmonMetrics`)."""

from typing import Self

from pydantic import BaseModel

from exo.shared.types.profiling import MemoryUsage, SystemPerformanceProfile
from exo.utils.pydantic_ext import TaggedModel


class _TempMetrics(BaseModel, extra="ignore"):
    """Temperature-related metrics returned by macmon."""

    cpu_temp_avg: float
    gpu_temp_avg: float


class _MemoryMetrics(BaseModel, extra="ignore"):
    """Memory-related metrics returned by macmon."""

    ram_total: int
    ram_usage: int
    swap_total: int
    swap_usage: int


class RawMacmonMetrics(BaseModel, extra="ignore"):
    """Complete set of metrics returned by macmon.

    Unknown fields are ignored for forward-compatibility.
    """

    timestamp: str  # ignored; macmon sample time. No consumer uses it for
    # staleness checks yet — wired through if a stalled-pipe detector appears.
    temp: _TempMetrics
    memory: _MemoryMetrics
    ecpu_usage: tuple[int, float]  # freq mhz, usage %
    pcpu_usage: tuple[int, float]  # freq mhz, usage %
    gpu_usage: tuple[int, float]  # freq mhz, usage %
    all_power: float
    ane_power: float
    cpu_power: float
    gpu_power: float
    gpu_ram_power: float
    ram_power: float
    sys_power: float


# macmon reports usage as FRACTIONS in [0, 1] (e.g. 0.9947 = 99.47%),
# not percentages. Consumers that display percents multiply by 100
# (e.g. EXO's NodeViewModel: ``gpuUsage * 100``). Clamp at the adapter
# boundary so transient out-of-range samples (a busy/idle race can emit
# a value slightly >1.0) never propagate a bogus 350% GPU reading.
_USAGE_FRACTION_RANGE = (0.0, 1.0)


class MacmonMetrics(TaggedModel):
    system_profile: SystemPerformanceProfile
    memory: MemoryUsage

    @staticmethod
    def _clamp_usage(value: float) -> float:
        """Clamp a macmon usage fraction to the [0, 1] range."""
        low, high = _USAGE_FRACTION_RANGE
        return max(low, min(high, value))

    @classmethod
    def from_raw(cls, raw: RawMacmonMetrics) -> Self:
        # NOTE: cpu_temp_avg and the sub-power breakdown (all_power,
        # ane_power, cpu_power, gpu_power, gpu_ram_power, ram_power) are
        # parsed but INTENTIONALLY dropped here — no consumer of
        # SystemPerformanceProfile surface (Swift decodes gpuUsage/temp/
        # sysPower/pcpuUsage/ecpuUsage only; dashboard reads the same five)
        # needs them yet. They remain in RawMacmonMetrics so a future
        # profile field (e.g. TODO flops_fp16 / ane_power) can map without
        # a macmon schema change.
        return cls(
            system_profile=SystemPerformanceProfile(
                gpu_usage=cls._clamp_usage(raw.gpu_usage[1]),
                temp=raw.temp.gpu_temp_avg,
                sys_power=raw.sys_power,
                pcpu_usage=cls._clamp_usage(raw.pcpu_usage[1]),
                ecpu_usage=cls._clamp_usage(raw.ecpu_usage[1]),
            ),
            memory=MemoryUsage.from_bytes(
                ram_total=raw.memory.ram_total,
                ram_available=(raw.memory.ram_total - raw.memory.ram_usage),
                swap_total=raw.memory.swap_total,
                swap_available=raw.memory.swap_total - raw.memory.swap_usage,
            ),
        )

    @classmethod
    def from_raw_json(cls, json: str) -> Self:
        """Parse a SINGLE line of macmon ``pipe`` output.

        macmon emits one JSON object per line; callers that stream
        ``pipe`` output must split on newlines and hand each non-blank
        line to this method (see InfoGatherer._monitor_macmon).
        """
        return cls.from_raw(RawMacmonMetrics.model_validate_json(json))
