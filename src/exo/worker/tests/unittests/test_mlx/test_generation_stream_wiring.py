"""Constructing an MlxBuilder is what applies the generation stream.

Upstream's two fast tests cover the two halves they can reach directly: that
`exo.worker.engines.mlx.generator.generate` re-exports mlx-lm's generation stream, and
that `use_generation_stream_by_default()` makes it the thread's default stream. Neither
exercises the WIRING -- that a runner reaching MLX actually gets there through
`MlxBuilder.__post_init__`, which is the only thing that calls the helper on the runner's
thread. Neutering `__post_init__` leaves both upstream tests passing.

So this checks the construction path itself: build an MlxBuilder the way `bootstrap.py`
does and assert the thread's default GPU stream is mlx-lm's generation stream afterwards,
while the original default stream is restored on exit so no other test inherits it.

mlx-lm's `generation_stream` is a `ThreadLocalStream`, which never compares equal to the
concrete `Stream` that becomes the thread default -- entering it resolves to this thread's
stream. So the expectation is resolved the same way `use_generation_stream_by_default`
does, by reading the default stream inside a `mx.stream(generation_stream)` context.
"""

import mlx.core as mx
import pytest
from mlx_lm.generate import generation_stream as mlx_lm_generation_stream

from exo.worker.engines.mlx.builder import MlxBuilder


def test_building_an_mlx_builder_applies_the_generation_stream() -> None:
    before = mx.default_stream(mx.default_device())
    with mx.stream(mlx_lm_generation_stream):
        generation = mx.default_stream(mx.default_device())
    try:
        if before == generation:
            pytest.skip("generation stream already the default before construction")

        # bootstrap.py: MlxBuilder(model_id, event_sender, cancel_receiver). The
        # constructor never touches `self.group` (only connect() does), so the
        # collaborators can be sentinels here.
        MlxBuilder("some/model", object(), object())  # pyright: ignore[reportArgumentType]

        assert mx.default_stream(mx.default_device()) == generation
    finally:
        mx.set_default_stream(before)