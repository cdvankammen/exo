"""Optional Prometheus metrics for exo.

Enable by setting ``EXO_METRICS_PORT`` environment variable.
Requires the ``prometheus-client`` package (``pip install prometheus-client``).
Falls back gracefully to a no-op if the package is missing.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from exo.api.types import GenerationStats

EXO_METRICS_PORT: str | None = os.getenv("EXO_METRICS_PORT")


def _has_prometheus() -> bool:
    try:
        import prometheus_client  # noqa: F401

        return True
    except ImportError:
        return False


# Lazy-loaded Prometheus objects — only initialised when the port is set.
_registry = None
ttft_histogram = None
latency_histogram = None
tokens_per_second_histogram = None
decode_latency_histogram = None
total_requests_counter = None
_init_lock = threading.Lock()
_initialised = False


def init_metrics() -> None:
    """Start the Prometheus HTTP server (once) if configured."""
    global _registry, _initialised
    global ttft_histogram, latency_histogram, tokens_per_second_histogram
    global decode_latency_histogram, total_requests_counter

    with _init_lock:
        if _initialised:
            return
        if EXO_METRICS_PORT is None:
            return
        if not _has_prometheus():
            logger.warning(
                "EXO_METRICS_PORT set but prometheus-client not installed — "
                "metrics disabled"
            )
            return

        from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

        _registry = CollectorRegistry()

        ttft_histogram = Histogram(
            "exo_time_to_first_token_seconds",
            "Time to first token (API-level)",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=_registry,
        )
        latency_histogram = Histogram(
            "exo_generation_duration_seconds",
            "Total generation wall-time",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=_registry,
        )
        tokens_per_second_histogram = Histogram(
            "exo_tokens_per_second",
            "Overall tokens per second",
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
            registry=_registry,
        )
        decode_latency_histogram = Histogram(
            "exo_decode_latency_seconds",
            "Engine-level decode latency (first token to last token)",
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=_registry,
        )
        total_requests_counter = Counter(
            "exo_requests_total",
            "Total generation requests",
            registry=_registry,
        )

        port = int(EXO_METRICS_PORT)
        start_http_server(port, registry=_registry)
        logger.info(f"Prometheus metrics server started on port {port}")
        _initialised = True


def record_generation_stats(stats: GenerationStats | None) -> None:
    """Record GenerationStats into Prometheus histograms (no-op when disabled)."""
    if stats is None or total_requests_counter is None:
        return

    total_requests_counter.inc()

    if stats.time_to_first_token_ms is not None and ttft_histogram is not None:
        ttft_histogram.observe(stats.time_to_first_token_ms / 1000)

    if stats.total_time_ms is not None and latency_histogram is not None:
        latency_histogram.observe(stats.total_time_ms / 1000)

    if stats.tokens_per_second is not None and tokens_per_second_histogram is not None:
        tokens_per_second_histogram.observe(stats.tokens_per_second)

    if stats.decode_latency_ms is not None and decode_latency_histogram is not None:
        decode_latency_histogram.observe(stats.decode_latency_ms / 1000)
