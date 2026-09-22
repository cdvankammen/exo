"""Regression tests: auth-gated models must not spam download-status scans.

GLM-4.7-8bit-gs32 (and any gated repo) requires HF_TOKEN.  The periodic
download-status scan iterates ALL known model cards, so a gated repo makes
``get_shard_download_status()`` raise ``HuggingFaceAuthenticationError`` on
every rescan.  Before the fix this:

  1. logged a misleading warning per scan tick ("Error downloading shard"),
  2. surfaced the auth error repeatedly in logs for ALL models,
  3. could downgrade/confuse the UI status of unrelated models.

The bug is cosmetic (inference is never blocked by auth), but the log spam
is real.  Fix: ``download_shard(skip_download=True)`` returns a
``not_started`` progress for auth-gated repos instead of raising, and
``get_shard_download_status()`` tolerates any residual auth error with an
informative info-level message.
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download.download_utils import (
    HuggingFaceAuthenticationError,
    download_shard,
)
from exo.download.impl_shard_downloader import ResumableShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata

AUTH_GATED_MODEL = ModelId("mlx-community/GLM-4.7-8bit-gs32")


def _make_shard(model_id: ModelId = AUTH_GATED_MODEL) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100),
            n_layers=91,
            hidden_size=5120,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=91,
        n_layers=91,
    )


@pytest.mark.asyncio
async def test_download_shard_status_check_tolerates_auth_error(tmp_path: Path) -> None:
    """skip_download=True (the status-check path) must NOT propagate auth errors.

    A gated repo raises HuggingFaceAuthenticationError via
    fetch_file_list_with_cache.  The status scan calls download_shard with
    skip_download=True; that path must yield a not_started progress instead
    of crashing the whole status sweep.
    """
    shard = _make_shard()

    async def _noop(
        _cb_shard: ShardMetadata, _progress: object
    ) -> None:
        return

    with patch(
        "exo.download.download_utils.fetch_file_list_with_cache",
        side_effect=HuggingFaceAuthenticationError(
            f"Model '{AUTH_GATED_MODEL}' requires authentication"
        ),
    ):
        path, progress = await download_shard(
            shard,
            _noop,
            skip_download=True,
            skip_internet=False,
        )

    assert progress.status == "not_started"
    assert progress.completed_files == 0
    assert progress.total_files == 0
    assert path == Path("models") / AUTH_GATED_MODEL.normalize() or path.name == (
        AUTH_GATED_MODEL.normalize()
    )


@pytest.mark.asyncio
async def test_download_shard_real_download_still_raises_auth_error(
    tmp_path: Path,
) -> None:
    """A REAL download of an auth-gated repo must still raise.

    The fix only suppresses auth errors on the status-check path
    (skip_download=True).  An actual download attempt must surface
    DownloadFailed with the HF_TOKEN guidance — swallowing it would hide a
    genuine user-facing error.
    """
    shard = _make_shard()

    async def _noop(
        _cb_shard: ShardMetadata, _progress: object
    ) -> None:
        return

    with (
        patch(
            "exo.download.download_utils.fetch_file_list_with_cache",
            side_effect=HuggingFaceAuthenticationError(
                f"Model '{AUTH_GATED_MODEL}' requires authentication"
            ),
        ),
        pytest.raises(HuggingFaceAuthenticationError),
    ):
        await download_shard(
            shard,
            _noop,
            skip_download=False,
            skip_internet=False,
        )


@pytest.mark.asyncio
async def test_get_shard_download_status_skips_auth_gated_models() -> None:
    """get_shard_download_status() must yield nothing (not raise) when the
    only known model is auth-gated."""

    class _AuthGatedCardCache:
        async def list_all(self) -> list[ModelCard]:
            return [
                ModelCard(
                    model_id=AUTH_GATED_MODEL,
                    storage_size=Memory.from_mb(100),
                    n_layers=91,
                    hidden_size=5120,
                    supports_tensor=False,
                    tasks=[ModelTask.TextGeneration],
                    backends=[Backend.MlxMetal],
                )
            ]

    downloader = ResumableShardDownloader(max_parallel_downloads=4)

    with (
        patch(
            "exo.download.download_utils.fetch_file_list_with_cache",
            side_effect=HuggingFaceAuthenticationError(
                f"Model '{AUTH_GATED_MODEL}' requires authentication"
            ),
        ),
        patch(
            "exo.download.impl_shard_downloader.model_cards.card_cache",
            _AuthGatedCardCache(),
        ),
    ):
        results: list[tuple[Path, object]] = []
        async for path, progress in downloader.get_shard_download_status():
            results.append((path, progress))

    # The auth-gated model is skipped, but the sweep itself completes.
    assert results == []