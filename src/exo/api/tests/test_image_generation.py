# pyright: reportUnusedFunction=false
"""Tests for image generation lifecycle: creation, polling, completion, failure, concurrency."""

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import API, _ensure_seed
from exo.api.types import AdvancedImageParams, ImageGenerationTaskParams
from exo.shared.types.chunks import ErrorChunk, ImageChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import ShardAssignments
from exo.utils.channels import channel

_TEST_MODEL = ModelId("test-img-model")


def _make_api() -> Any:
    """Create a minimal API instance with image generation route and error handler."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    api.state = State()
    api.port = 52415
    api._sent_image_hashes = set()  # pyright: ignore[reportPrivateUsage]
    app.post("/v1/images/generations", response_model=None)(api.image_generations)
    return api


def _add_model_instance(api: Any, model_id: str = "test-img-model") -> None:
    """Register a fake instance so model validation passes."""
    inst_id = InstanceId(f"inst-{model_id}")
    assignments = MagicMock(spec=ShardAssignments)
    assignments.model_id = ModelId(model_id)
    instance = MagicMock(spec=Instance)
    instance.shard_assignments = assignments
    api.state.instances[inst_id] = instance


# ---------------------------------------------------------------------------
# 1. Request shape & command creation
# ---------------------------------------------------------------------------


def test_image_generation_request_sends_command_with_correct_params() -> None:
    """POST /v1/images/generations dispatches an ImageGeneration command.

    We mock _collect_image_generation to avoid blocking on chunk collection,
    then verify the command that was sent via _send.
    """
    api = _make_api()
    _add_model_instance(api)
    client = TestClient(api.app)

    # Patch the collector so the endpoint returns immediately
    async def _fake_collect(*args: Any, **kwargs: Any) -> Any:
        from exo.api.types import ImageData, ImageGenerationResponse

        return ImageGenerationResponse(data=[ImageData(b64_json="dGVzdA==")])

    api._collect_image_generation = _fake_collect  # pyright: ignore[reportPrivateUsage]

    response = client.post(
        "/v1/images/generations",
        json={"prompt": "a sunset", "model": "test-img-model", "size": "1024x1024"},
    )
    assert response.status_code == 200

    # Verify _send was called with an ImageGeneration command
    assert api._send.called  # pyright: ignore[reportPrivateUsage]
    cmd = api._send.call_args_list[0][0][0]  # pyright: ignore[reportPrivateUsage]
    assert hasattr(cmd, "task_params")
    assert cmd.task_params.prompt == "a sunset"
    assert cmd.task_params.model == "test-img-model"
    assert cmd.task_params.size == "1024x1024"


# ---------------------------------------------------------------------------
# 2. Model validation (missing model → 404)
# ---------------------------------------------------------------------------


def test_image_generation_missing_model_returns_404() -> None:
    """Requesting generation for a model with no instance returns 404."""
    api = _make_api()
    # No instance registered
    client = TestClient(api.app)

    response = client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "model": "nonexistent-model"},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 3. Invalid parameters (bad size → 422)
# ---------------------------------------------------------------------------


def test_image_generation_invalid_size_returns_422() -> None:
    """Invalid size parameter is rejected by Pydantic validation."""
    api = _make_api()
    _add_model_instance(api)
    client = TestClient(api.app)

    response = client.post(
        "/v1/images/generations",
        json={"prompt": "a dog", "model": "test-img-model", "size": "9999x9999"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 4. Seed handling (_ensure_seed helper)
# ---------------------------------------------------------------------------


def test_ensure_seed_assigns_seed_when_none() -> None:
    """When advanced_params is None, _ensure_seed creates one with a valid seed."""
    result = _ensure_seed(None)
    assert isinstance(result, AdvancedImageParams)
    assert result.seed is not None
    assert result.seed >= 0


def test_ensure_seed_preserves_existing_seed() -> None:
    """When advanced_params already has a seed, it is not overwritten."""
    params = AdvancedImageParams(seed=42)
    result = _ensure_seed(params)
    assert result.seed == 42


def test_ensure_seed_fills_missing_seed_on_existing_params() -> None:
    """When advanced_params exists but seed is None, a seed is assigned."""
    params = AdvancedImageParams(num_inference_steps=20)
    result = _ensure_seed(params)
    assert result.seed is not None
    assert result.num_inference_steps == 20


# ---------------------------------------------------------------------------
# 5. Chunk collection — successful completion
# ---------------------------------------------------------------------------


def test_collect_image_chunks_assembles_full_image() -> None:
    """Two chunks for one image are reassembled into the correct b64 payload."""
    api = _make_api()
    cid = CommandId("img-collect-ok")

    part1 = base64.b64encode(b"\x89PNG\r\n").decode()
    part2 = base64.b64encode(b"\x1a\n").decode()

    async def _run():  # noqa: ANN202
        async def collector():  # noqa: ANN202
            return await api._collect_image_chunks(  # pyright: ignore[reportPrivateUsage]
                request=None, command_id=cid, num_images=1, response_format="b64_json", capture_stats=False,
            )

        task = asyncio.create_task(collector())
        # Wait until collector creates the queue
        for _ in range(100):
            await asyncio.sleep(0.01)
            if cid in api._image_generation_queues:  # pyright: ignore[reportPrivateUsage]
                break
        else:
            raise AssertionError("collector never created queue")
        sender = api._image_generation_queues[cid]  # pyright: ignore[reportPrivateUsage]
        await sender.send(
            ImageChunk(model=_TEST_MODEL, data=part1, chunk_index=0, total_chunks=2, image_index=0, format="png")
        )
        await sender.send(
            ImageChunk(model=_TEST_MODEL, data=part2, chunk_index=1, total_chunks=2, image_index=0, format="png")
        )
        sender.close()
        return await task

    images, stats = asyncio.run(_run())
    assert len(images) == 1
    assert images[0].b64_json == part1 + part2
    assert stats is None


# ---------------------------------------------------------------------------
# 6. Chunk collection — error chunk raises 500
# ---------------------------------------------------------------------------


def test_collect_image_chunks_error_raises_500() -> None:
    """An ErrorChunk during collection raises HTTPException with status 500."""
    api = _make_api()
    cid = CommandId("img-collect-err")

    async def _run():  # noqa: ANN202
        async def collector():  # noqa: ANN202
            return await api._collect_image_chunks(  # pyright: ignore[reportPrivateUsage]
                request=None, command_id=cid, num_images=1, response_format="b64_json", capture_stats=False,
            )

        task = asyncio.create_task(collector())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if cid in api._image_generation_queues:  # pyright: ignore[reportPrivateUsage]
                break
        else:
            raise AssertionError("collector never created queue")
        sender = api._image_generation_queues[cid]  # pyright: ignore[reportPrivateUsage]
        await sender.send(ErrorChunk(model=_TEST_MODEL, error_message="GPU OOM during generation"))
        sender.close()
        return await task

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_run())
    assert exc_info.value.status_code == 500
    assert "GPU OOM" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# 7. Cancel active image generation
# ---------------------------------------------------------------------------


def test_cancel_active_image_generation_closes_sender() -> None:
    """Cancelling an active image gen closes its sender and sends TaskCancelled."""
    api = _make_api()
    api.app.post("/v1/cancel/{command_id}")(api.cancel_command)
    client = TestClient(api.app)

    cid = CommandId("img-cancel-789")
    sender = MagicMock()
    api._image_generation_queues[cid] = sender  # pyright: ignore[reportPrivateUsage]

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()  # pyright: ignore[reportPrivateUsage]
    task_cancelled = api._send.call_args[0][0]  # pyright: ignore[reportPrivateUsage]
    assert task_cancelled.cancelled_command_id == cid

# ---------------------------------------------------------------------------
# 8. Concurrent generations use separate queues
# ---------------------------------------------------------------------------


def test_concurrent_image_generations_use_separate_queues() -> None:
    """Two concurrent generations each get an independent queue and both complete."""
    api = _make_api()
    cid_a = CommandId("img-concurrent-a")
    cid_b = CommandId("img-concurrent-b")
    part = base64.b64encode(b"fake").decode()

    async def _run():  # noqa: ANN202
        async def collect_a():  # noqa: ANN202
            return await api._collect_image_chunks(  # pyright: ignore[reportPrivateUsage]
                request=None, command_id=cid_a, num_images=1, response_format="b64_json", capture_stats=False,
            )

        async def collect_b():  # noqa: ANN202
            return await api._collect_image_chunks(  # pyright: ignore[reportPrivateUsage]
                request=None, command_id=cid_b, num_images=1, response_format="b64_json", capture_stats=False,
            )

        task_a = asyncio.create_task(collect_a())
        task_b = asyncio.create_task(collect_b())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if cid_a in api._image_generation_queues and cid_b in api._image_generation_queues:  # pyright: ignore[reportPrivateUsage]
                break
        else:
            raise AssertionError("collectors never created queues")
        sender_a = api._image_generation_queues[cid_a]  # pyright: ignore[reportPrivateUsage]
        sender_b = api._image_generation_queues[cid_b]  # pyright: ignore[reportPrivateUsage]
        # Ensure they are distinct objects
        assert sender_a is not sender_b
        await sender_a.send(ImageChunk(model=_TEST_MODEL, data=part, chunk_index=0, total_chunks=1, image_index=0, format="png"))
        await sender_b.send(ImageChunk(model=_TEST_MODEL, data=part, chunk_index=0, total_chunks=1, image_index=0, format="png"))
        sender_a.close()
        sender_b.close()
        images_a, _ = await task_a
        images_b, _ = await task_b
        return images_a, images_b

    images_a, images_b = asyncio.run(_run())
    assert len(images_a) == 1
    assert len(images_b) == 1
    assert images_a[0].b64_json == part
    assert images_b[0].b64_json == part
    # Queues must be cleaned up after completion
    assert cid_a not in api._image_generation_queues  # pyright: ignore[reportPrivateUsage]
    assert cid_b not in api._image_generation_queues  # pyright: ignore[reportPrivateUsage]
