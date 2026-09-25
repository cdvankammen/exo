"""Zero-dependency Prometheus metrics for exo.

Implements the Prometheus text exposition format (``text/plain; version=0.0.4``)
with counters, gauges, and histograms — no external ``prometheus_client``
dependency required. All metrics are process-local and registered in a shared
registry so multiple API instances (one per GPU host) can each export their own
``/metrics`` endpoint.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Iterable, TypeVar

_MetricT = TypeVar("_MetricT", bound="Counter | Gauge | Histogram")


class Counter:
    """Monotonically increasing counter keyed by label values."""

    __slots__ = ("name", "help", "_labelnames", "_values", "_lock")

    def __init__(
        self, name: str, documentation: str, labelnames: Iterable[str] = ()
    ) -> None:
        self.name = name
        self.help = documentation
        self._labelnames = tuple(labelnames)
        self._values: dict[tuple[str, ...], float] = OrderedDict()
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str] | None) -> tuple[str, ...]:
        labels = labels or {}
        return tuple(labels.get(name, "") for name in self._labelnames)

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            items = list(self._values.items())
        for key, val in items:
            label_str = ",".join(
                f'{n}="{v}"' for n, v in zip(self._labelnames, key, strict=True)
            )
            suffix = f"{{{label_str}}}" if label_str else ""
            lines.append(f"{self.name}{suffix} {val}")
        return "\n".join(lines) + "\n"


class Gauge:
    """Current-value metric keyed by label values."""

    __slots__ = ("name", "help", "_labelnames", "_values", "_lock")

    def __init__(
        self, name: str, documentation: str, labelnames: Iterable[str] = ()
    ) -> None:
        self.name = name
        self.help = documentation
        self._labelnames = tuple(labelnames)
        self._values: dict[tuple[str, ...], float] = OrderedDict()
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str] | None) -> tuple[str, ...]:
        labels = labels or {}
        return tuple(labels.get(name, "") for name in self._labelnames)

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - value

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            items = list(self._values.items())
        for key, val in items:
            label_str = ",".join(
                f'{n}="{v}"' for n, v in zip(self._labelnames, key, strict=True)
            )
            suffix = f"{{{label_str}}}" if label_str else ""
            lines.append(f"{self.name}{suffix} {val}")
        return "\n".join(lines) + "\n"


_DEFAULT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
    0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
)


class Histogram:
    """Histogram of observations bucketed by value.

    Stores bucket counts, sum, and count in a ``_HistogramEntry`` for clean
    type narrowing. Renders ``_bucket`` (cumulative), ``_sum`` and ``_count``
    so the Prometheus scraper can compute quantiles.
    """

    __slots__ = ("name", "help", "_labelnames", "_buckets", "_entries", "_lock")

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Iterable[str] = (),
        buckets: Iterable[float] = _DEFAULT_BUCKETS,
    ) -> None:
        self.name = name
        self.help = documentation
        self._labelnames = tuple(labelnames)
        self._buckets = tuple(buckets)
        self._entries: dict[tuple[str, ...], list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str] | None) -> tuple[str, ...]:
        labels = labels or {}
        return tuple(labels.get(name, "") for name in self._labelnames)

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                # [bucket_0, ..., bucket_n, sum, count]
                entry = [0.0] * (len(self._buckets) + 2)
                self._entries[key] = entry
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    entry[i] += 1.0
            entry[len(self._buckets)] += value  # sum
            entry[len(self._buckets) + 1] += 1.0  # count

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            items = list(self._entries.items())
        for key, entry in items:
            label_parts = ",".join(
                f'{n}="{v}"' for n, v in zip(self._labelnames, key, strict=True)
            )
            # buckets
            for i, bound in enumerate(self._buckets):
                le_label = f'le="{bound}"'
                all_labels = f"{label_parts},{le_label}" if label_parts else le_label
                lines.append(f"{self.name}_bucket{{{all_labels}}} {entry[i]}")
            # +Inf bucket = cumulative count of ALL observations
            inf_tag = f"{label_parts},le=\"+Inf\"" if label_parts else "le=\"+Inf\""
            lines.append(f"{self.name}_bucket{{{inf_tag}}} {entry[len(self._buckets) + 1]}")
            # sum + count
            suffix = f"{{{label_parts}}}" if label_parts else ""
            lines.append(f"{self.name}_sum{suffix} {entry[len(self._buckets)]}")
            lines.append(f"{self.name}_count{suffix} {entry[len(self._buckets) + 1]}")
        return "\n".join(lines) + "\n"


class MetricsRegistry:
    """Holds all metric families and renders the combined exposition."""

    __slots__ = ("_metrics", "_lock")

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = OrderedDict()
        self._lock = threading.Lock()

    def register(self, metric: _MetricT) -> _MetricT:
        with self._lock:
            if metric.name in self._metrics:
                return self._metrics[metric.name]  # type: ignore[return-value]
            self._metrics[metric.name] = metric
            return metric

    def render(self) -> str:
        with self._lock:
            metrics = list(self._metrics.values())
        return "".join(m.render() for m in metrics)


# --- Shared process-wide registry --------------------------------------------------

_REGISTRY = MetricsRegistry()


def register(metric: _MetricT) -> _MetricT:
    return _REGISTRY.register(metric)


def render_all() -> str:
    return _REGISTRY.render()


# --- Metric family singletons ------------------------------------------------------

REQUEST_COUNT = register(
    Counter("exo_requests_total", "Total API requests processed", ["model", "endpoint", "status"])
)

REQUEST_LATENCY = register(
    Histogram("exo_request_latency_seconds", "End-to-end request latency (seconds)", ["model", "endpoint"])
)

TTFT = register(
    Histogram("exo_time_to_first_token_seconds", "Time to first token (seconds)", ["model", "endpoint"])
)

TOKENS_PER_SECOND = register(
    Histogram("exo_tokens_per_second", "Overall generation throughput (tok/s)", ["model", "endpoint"])
)

DECODE_TOKENS_PER_SECOND = register(
    Histogram("exo_decode_tokens_per_second", "Decode-phase throughput (tok/s)", ["model", "endpoint"])
)

GENERATION_TOKENS = register(
    Counter("exo_generation_tokens_total", "Total completion tokens generated", ["model", "endpoint"])
)

IN_FLIGHT_REQUESTS = register(
    Gauge("exo_in_flight_requests", "Number of requests being processed", ["model", "endpoint"])
)

ERRORS_TOTAL = register(
    Counter("exo_errors_total", "Total API errors by type", ["model", "endpoint", "error_type"])
)

GENERATION_STATS_TOTAL = register(
    Counter("exo_generation_stats_total", "Completed generation runs", ["model"])
)


def record_request(
    *,
    model: str,
    endpoint: str,
    status: str,
    latency_seconds: float,
    ttft_seconds: float | None = None,
    tokens_per_second: float | None = None,
    decode_tokens_per_second: float | None = None,
    generation_tokens: int | None = None,
    error_type: str | None = None,
) -> None:
    """Record one completed API request and all associated latency metrics.

    Call exactly once per request after the response has been fully generated
    (or errored). Safe to call from any thread.
    """
    labels = {"model": model, "endpoint": endpoint}
    REQUEST_COUNT.inc(1.0, {**labels, "status": status})
    REQUEST_LATENCY.observe(latency_seconds, labels)
    if ttft_seconds is not None:
        TTFT.observe(ttft_seconds, labels)
    if tokens_per_second is not None and tokens_per_second > 0:
        TOKENS_PER_SECOND.observe(tokens_per_second, labels)
    if decode_tokens_per_second is not None and decode_tokens_per_second > 0:
        DECODE_TOKENS_PER_SECOND.observe(decode_tokens_per_second, labels)
    if generation_tokens is not None and generation_tokens > 0:
        GENERATION_TOKENS.inc(float(generation_tokens), labels)
    if error_type is not None:
        ERRORS_TOTAL.inc(1.0, {**labels, "error_type": error_type})


def record_generation_stats(model: str) -> None:
    """Increment the generation-stats counter for a completed generation run."""
    GENERATION_STATS_TOTAL.inc(1.0, {"model": model})


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "record_request",
    "record_generation_stats",
    "render_all",
]
