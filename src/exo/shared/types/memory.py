"""Memory value type with arithmetic helpers.

:class:`Memory` wraps RAM quantities with conversion/arithmetic methods used by placement and profiling."""

import os
from math import ceil
from typing import Self, overload

from exo.utils.pydantic_ext import FrozenModel


class Memory(FrozenModel):
    in_bytes: int = 0

    @classmethod
    def from_bytes(cls, val: int) -> Self:
        """Construct a new Memory object from a number of bytes"""
        return cls(in_bytes=val)

    @property
    def in_kb(self) -> int:
        """The approximate kilobytes this memory represents, rounded up. Setting this property rounds to the nearest byte."""
        return ceil(self.in_bytes / 1024)

    @in_kb.setter
    def in_kb(self, val: int):
        """Set this memorys value in kilobytes."""
        self.in_bytes = val * 1024

    @classmethod
    def from_kb(cls, val: int) -> Self:
        """Construct a new Memory object from a number of kilobytes"""
        return cls(in_bytes=val * 1024)

    @classmethod
    def from_float_kb(cls, val: float) -> Self:
        """Construct a new Memory object from a number of kilobytes, rounding where appropriate"""
        return cls(in_bytes=round(val * 1024))

    @property
    def in_mb(self) -> int:
        """The approximate megabytes this memory represents, rounded to nearest MB. Setting this property rounds to the nearest byte."""
        return round(self.in_bytes / (1024**2))

    @in_mb.setter
    def in_mb(self, val: int):
        """Set the megabytes for this memory."""
        self.in_bytes = val * (1024**2)

    @property
    def in_float_mb(self) -> float:
        """The megabytes this memory represents as a float. Setting this property rounds to the nearest byte."""
        return self.in_bytes / (1024**2)

    @in_float_mb.setter
    def in_float_mb(self, val: float):
        """Set the megabytes for this memory, rounded to the nearest byte."""
        self.in_bytes = round(val * (1024**2))

    @classmethod
    def from_mb(cls, val: float) -> Self:
        """Construct a new Memory object from a number of megabytes"""
        return cls(in_bytes=round(val * (1024**2)))

    @classmethod
    def from_gb(cls, val: float) -> Self:
        """Construct a new Memory object from a number of megabytes"""
        return cls(in_bytes=round(val * (1024**3)))

    @property
    def in_gb(self) -> float:
        """The approximate gigabytes this memory represents."""
        return self.in_bytes / (1024**3)

    def __add__(self, other: object) -> "Memory":
        if isinstance(other, Memory):
            return Memory.from_bytes(self.in_bytes + other.in_bytes)
        return NotImplemented

    def __radd__(self, other: object) -> "Memory":
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: object) -> "Memory":
        if isinstance(other, Memory):
            return Memory.from_bytes(self.in_bytes - other.in_bytes)
        return NotImplemented

    def __mul__(self, other: int | float):
        return Memory.from_bytes(round(self.in_bytes * other))

    def __rmul__(self, other: int | float):
        return self * other

    @overload
    def __truediv__(self, other: "Memory") -> float: ...
    @overload
    def __truediv__(self, other: int) -> "Memory": ...
    @overload
    def __truediv__(self, other: float) -> "Memory": ...
    def __truediv__(self, other: object) -> "Memory | float":
        if isinstance(other, Memory):
            return self.in_bytes / other.in_bytes
        if isinstance(other, (int, float)):
            return Memory.from_bytes(round(self.in_bytes / other))
        return NotImplemented

    def __floordiv__(self, other: object) -> "Memory":
        if isinstance(other, (int, float)):
            return Memory.from_bytes(int(self.in_bytes // other))
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Memory):
            return self.in_bytes < other.in_bytes
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, Memory):
            return self.in_bytes <= other.in_bytes
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Memory):
            return self.in_bytes > other.in_bytes
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, Memory):
            return self.in_bytes >= other.in_bytes
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Memory):
            return self.in_bytes == other.in_bytes
        return NotImplemented

    def __repr__(self) -> str:
        return f"Memory.from_bytes({self.in_bytes})"

    def __str__(self) -> str:
        if self.in_gb > 2:
            val = self.in_gb
            unit = "GiB"
        elif self.in_mb > 2:
            val = self.in_mb
            unit = "MiB"
        elif self.in_kb > 3:
            val = self.in_kb
            unit = "KiB"
        else:
            val = self.in_bytes
            unit = "B"

        return f"{val:.2f} {unit}".rstrip("0").rstrip(".") + f" {unit}"


# ---------------------------------------------------------------------------
# Memory-pressure monitoring for OOM prevention (exo-explore/exo#1626).
# Uses the Darwin-safe virtual_memory_statistics() helper, never raw psutil,
# so these work on macOS 26 / Darwin 27 hosts.
# ---------------------------------------------------------------------------

def get_system_memory_total() -> Memory:
    """Total system memory."""
    from exo.utils.virtual_memory import virtual_memory_statistics

    return Memory.from_bytes(virtual_memory_statistics().total_bytes)


def get_system_memory_available() -> Memory:
    """Currently available system memory."""
    from exo.utils.virtual_memory import virtual_memory_statistics

    return Memory.from_bytes(virtual_memory_statistics().available_bytes)


def get_system_memory_used() -> Memory:
    """Currently used system memory."""
    from exo.utils.virtual_memory import virtual_memory_statistics

    stats = virtual_memory_statistics()
    return Memory.from_bytes(stats.total_bytes - stats.available_bytes)


def get_memory_pressure_percent() -> float:
    """Current memory pressure as a percentage (0-100)."""
    from exo.utils.virtual_memory import virtual_memory_statistics

    stats = virtual_memory_statistics()
    return stats.used_fraction * 100.0


def is_memory_pressure_high(threshold_percent: float | None = None) -> bool:
    """True when current memory pressure is at/above the threshold."""
    if threshold_percent is None:
        threshold_percent = MEMORY_PRESSURE_THRESHOLD_PERCENT
    return get_memory_pressure_percent() >= threshold_percent


# Default memory pressure threshold for OOM prevention (90% used).
MEMORY_PRESSURE_THRESHOLD_PERCENT = float(
    os.environ.get("EXO_MEMORY_PRESSURE_THRESHOLD", "90.0")
)
