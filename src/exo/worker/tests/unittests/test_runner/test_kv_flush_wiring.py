# type: ignore
"""Runner wiring: the idle/shutdown KV flush reaches the engine's prefix cache."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from exo.worker.runner.runner import Runner


def test_flush_calls_engine_prefix_cache():
    kv = MagicMock()
    fake_self = SimpleNamespace(generator=SimpleNamespace(kv_prefix_cache=kv))
    Runner._flush_kv_cache_to_disk(fake_self)
    kv.flush_to_disk.assert_called_once_with(force=True)


def test_flush_noop_for_engine_without_cache():
    # Builder before build(), or engines without a prefix cache (e.g. image).
    fake_self = SimpleNamespace(generator=object())
    Runner._flush_kv_cache_to_disk(fake_self)  # must not raise


def test_flush_noop_when_cache_is_none():
    fake_self = SimpleNamespace(generator=SimpleNamespace(kv_prefix_cache=None))
    Runner._flush_kv_cache_to_disk(fake_self)  # must not raise


def test_periodic_cleanup_calls_cache():
    kv = MagicMock()
    fake_self = SimpleNamespace(generator=SimpleNamespace(kv_prefix_cache=kv))
    Runner._periodic_kv_cleanup(fake_self)
    kv._periodic_cleanup.assert_called_once_with()


def test_periodic_cleanup_noop_for_engine_without_cache():
    # Builder before build(), or engines without a prefix cache (e.g. image).
    fake_self = SimpleNamespace(generator=object())
    Runner._periodic_kv_cleanup(fake_self)  # must not raise


def test_periodic_cleanup_noop_when_cache_is_none():
    fake_self = SimpleNamespace(generator=SimpleNamespace(kv_prefix_cache=None))
    Runner._periodic_kv_cleanup(fake_self)  # must not raise


def test_periodic_cleanup_noop_when_cache_lacks_method():
    # A cache-like object that merely lacks _periodic_cleanup (older engine
    # with a prefix cache attr but no cleanup support) must be tolerated.
    kv = object()
    fake_self = SimpleNamespace(generator=SimpleNamespace(kv_prefix_cache=kv))
    Runner._periodic_kv_cleanup(fake_self)  # must not raise


def test_periodic_cleanup_survives_cache_exception():
    kv = MagicMock()
    kv._periodic_cleanup.side_effect = RuntimeError("boom")
    fake_self = SimpleNamespace(generator=SimpleNamespace(kv_prefix_cache=kv))
    Runner._periodic_kv_cleanup(fake_self)  # must log-warn, not raise
