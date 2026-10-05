"""H3 test: generate.prefill waits for the in-flight decode step before its collectives.

Drives the REAL prefill() with stream_generate stubbed out, so the assertion is on the
ordering the port introduces: mx.synchronize(generation_stream) must happen AFTER
set_pipeline_prefill and BEFORE mx_barrier(group).

The model stub needs a `.layers` list (set_pipeline_prefill, set_pipeline_queue_sends and
_has_pipeline_communication_layer all walk it). Empty layers => not a pipeline model, so the
non-pipeline stream_generate branch runs, which is the branch that matters: the collectives
this port guards are exactly the ones between the wait and the prompt eval.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.generator import generate as generate_module
from exo.worker.engines.mlx.generator.generate import prefill


class _TrimmableCache:
    """Minimal KVCacheType stand-in: prefill() only iterates it and calls trim(2)."""

    def __init__(self) -> None:
        self.trimmed: list[int] = []

    def trim(self, n: int) -> None:
        self.trimmed.append(n)

    def __iter__(self):
        return iter(())


def test_prefill_waits_for_the_decode_step_before_the_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_synchronize(stream: object = None) -> None:
        assert stream is generate_module.generation_stream
        calls.append("wait")

    def fake_barrier(group: object = None) -> None:
        calls.append("barrier")

    def fake_set_pipeline_prefill(model: object, is_prefill: bool) -> None:
        calls.append(f"set_prefill {is_prefill}")

    def fake_set_pipeline_queue_sends(model: object, queue_sends: bool) -> None:
        calls.append(f"set_queue_sends {queue_sends}")

    def fake_stream_generate(**kwargs: object):
        # prefill() breaks after the first iteration; one yield is enough.
        calls.append("stream_generate")
        yield SimpleNamespace(token=mx.array([1]), text="x", logprobs=None)

    monkeypatch.setattr(mx, "synchronize", fake_synchronize)
    monkeypatch.setattr(generate_module, "mx_barrier", fake_barrier)
    monkeypatch.setattr(generate_module, "set_pipeline_prefill", fake_set_pipeline_prefill)
    monkeypatch.setattr(
        generate_module, "set_pipeline_queue_sends", fake_set_pipeline_queue_sends
    )
    monkeypatch.setattr(generate_module, "stream_generate", fake_stream_generate)

    model = SimpleNamespace(layers=[])
    cache = [_TrimmableCache()]

    _tok_per_sec, num_tokens, snapshots = prefill(
        model=model,  # pyright: ignore[reportArgumentType]
        tokenizer=SimpleNamespace(),  # pyright: ignore[reportArgumentType]
        sampler=lambda _logits: mx.array([1]),
        prompt_tokens=mx.array([1, 2, 3]),
        cache=cache,  # pyright: ignore[reportArgumentType]
        group=None,
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert num_tokens == 3
    assert snapshots == []

    # The wait must sit between the prefill flag and the barrier, and before the prompt eval.
    assert calls.index("set_prefill True") < calls.index("wait")
    assert calls.index("wait") < calls.index("barrier")
    assert calls.index("barrier") < calls.index("stream_generate")
    # And nothing waited before the flag was set: the decode step is only overlapped once
    # the model is flagged as prefilling, not earlier.
    assert calls[0] == "set_prefill True"


def test_prefill_does_not_wait_when_there_are_no_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 0-token early return must not pay for a stream wait."""
    calls: list[str] = []

    def fake_synchronize(stream: object = None) -> None:
        calls.append("wait")

    def fake_barrier(group: object = None) -> None:
        calls.append("barrier")

    monkeypatch.setattr(mx, "synchronize", fake_synchronize)
    monkeypatch.setattr(generate_module, "mx_barrier", fake_barrier)

    tok_per_sec, num_tokens, snapshots = prefill(
        model=SimpleNamespace(layers=[]),  # pyright: ignore[reportArgumentType]
        tokenizer=SimpleNamespace(),  # pyright: ignore[reportArgumentType]
        sampler=lambda _logits: mx.array([1]),
        prompt_tokens=mx.array([], dtype=mx.int32),
        cache=[],
        group=None,
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert (tok_per_sec, num_tokens, snapshots) == (0.0, 0, [])
    assert calls == []