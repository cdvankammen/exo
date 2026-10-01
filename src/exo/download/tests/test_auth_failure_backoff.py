"""Regression tests for #2376 — HF auth failures must not be re-probed forever.

The download coordinator rescans every model card every 60 s. Before this fix a
gated repo without a token (mlx-community/GLM-4.7-8bit-gs32) was re-fetched from
Hugging Face on every one of those scans and 401'd every time, all evening.
"""

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiofiles
import aiofiles.os as aios
import pytest

from exo.download.download_utils import (
    HuggingFaceAuthenticationError,
    fetch_file_list_with_cache,
)
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import FileListEntry


@pytest.fixture
def model_id() -> ModelId:
    return ModelId("mlx-community/GLM-4.7-8bit-gs32")


def _auth_fail_file(models_dir: Path, model_id: ModelId) -> Path:
    return (
        models_dir
        / "caches"
        / model_id.normalize()
        / f"{model_id.normalize()}--main--file_list.auth_fail.json"
    )


class TestAuthFailureBackoff:
    async def test_repeated_scan_does_not_repoll_hf(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """The 60 s status scan must contact HF once, not once per scan."""
        models_dir = tmp_path / "models"

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (models_dir,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", models_dir),
            patch(
                "exo.download.download_utils.get_hf_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "exo.download.download_utils.fetch_file_list_with_retry",
                new_callable=AsyncMock,
                side_effect=HuggingFaceAuthenticationError("401 gated repo"),
            ) as mock_fetch,
        ):
            # First scan: hits HF, fails auth, records the refusal.
            with pytest.raises(FileNotFoundError):
                await fetch_file_list_with_cache(model_id, "main")
            assert mock_fetch.await_count == 1
            assert await aios.path.exists(_auth_fail_file(models_dir, model_id))

            # Subsequent scans (the 60 s loop) must NOT hit HF again.
            for _ in range(5):
                with pytest.raises(FileNotFoundError):
                    await fetch_file_list_with_cache(model_id, "main")
            assert mock_fetch.await_count == 1, (
                "HF was re-polled on a later status scan; the auth failure was "
                "not cached"
            )

    async def test_backoff_expires_and_reprobes(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """After the TTL the repo is re-probed, so a later fix can take effect."""
        models_dir = tmp_path / "models"
        cache_dir = models_dir / "caches" / model_id.normalize()
        await aios.makedirs(cache_dir, exist_ok=True)
        auth_fail = _auth_fail_file(models_dir, model_id)

        # Backdate the refusal marker beyond the TTL.
        async with aiofiles.open(auth_fail, "w") as f:
            await f.write("{}")
        stale = time.time() - 60 * 60 - 5
        import os

        os.utime(auth_fail, (stale, stale))

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (models_dir,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", models_dir),
            patch(
                "exo.download.download_utils.get_hf_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "exo.download.download_utils.fetch_file_list_with_retry",
                new_callable=AsyncMock,
                side_effect=HuggingFaceAuthenticationError("401 gated repo"),
            ) as mock_fetch,
        ):
            with pytest.raises(FileNotFoundError):
                await fetch_file_list_with_cache(model_id, "main")
            assert mock_fetch.await_count == 1

    async def test_setting_a_token_retries_immediately(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """A token appearing after the failure re-probes at once."""
        models_dir = tmp_path / "models"
        cache_dir = models_dir / "caches" / model_id.normalize()
        await aios.makedirs(cache_dir, exist_ok=True)
        auth_fail = _auth_fail_file(models_dir, model_id)
        async with aiofiles.open(auth_fail, "w") as f:
            await f.write("{}")

        file_list = [FileListEntry(type="file", path="model.safetensors", size=10)]

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (models_dir,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", models_dir),
            patch(
                "exo.download.download_utils.get_hf_token",
                new_callable=AsyncMock,
                return_value="hf_someToken",
            ),
            patch(
                "exo.download.download_utils.fetch_file_list_with_retry",
                new_callable=AsyncMock,
                return_value=file_list,
            ) as mock_fetch,
        ):
            result = await fetch_file_list_with_cache(model_id, "main")

        assert result == file_list
        assert mock_fetch.await_count == 1
        # The stale refusal is cleared once HF answers successfully.
        assert not await aios.path.exists(auth_fail)

    async def test_successful_fetch_clears_a_prior_refusal(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        cache_dir = models_dir / "caches" / model_id.normalize()
        await aios.makedirs(cache_dir, exist_ok=True)
        auth_fail = _auth_fail_file(models_dir, model_id)
        async with aiofiles.open(auth_fail, "w") as f:
            await f.write("{}")

        file_list = [FileListEntry(type="file", path="config.json", size=3)]

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (models_dir,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", models_dir),
            patch(
                "exo.download.download_utils.get_hf_token",
                new_callable=AsyncMock,
                return_value="hf_someToken",
            ),
            patch(
                "exo.download.download_utils.fetch_file_list_with_retry",
                new_callable=AsyncMock,
                return_value=file_list,
            ),
        ):
            await fetch_file_list_with_cache(model_id, "main")

        assert not await aios.path.exists(auth_fail)
