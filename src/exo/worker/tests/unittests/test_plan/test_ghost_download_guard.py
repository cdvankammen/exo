"""Isolated regression test: ghost DownloadCompleted entries must not block a re-download.

When a DownloadCompleted event survives in the (replayed) event log but the
model directory was deleted / never fully written (e.g. a shard whose weights
failed a hash check and was rolled back, or a cancelled download whose stale
completion was never superseded), ``plan()`` must treat the model as ABSENT and
emit ``DownloadModel`` so the downloader can re-download.

This is the isolated test for the ghost-download guard in ``_model_needs_download``
(commit 32d9835d).  Unlike ``test_download_and_loading.py`` — whose autouse
fixture makes every DownloadCompleted trusted — this module exercises the
ghost path explicitly:

* no model dir on disk                          -> ghost -> DownloadModel
* model dir present but incomplete (resolve None) -> ghost -> DownloadModel
* model dir present AND complete (resolve Path)   -> NOT a ghost -> no DownloadModel

The model-dir existence check in ``_model_needs_download`` hits the REAL
filesystem (``(dir / model_id).is_dir()``), so this test uses a real
``tmp_path`` and only creates the model subdir when simulating a complete /
incomplete download.  ``resolve_existing_model`` is patched to keep the
completeness verdict deterministic.
"""

from pathlib import Path
from unittest.mock import patch

import exo.worker.plan as plan_mod
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerConnected
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def _plan_for_download(
    tmp_path: Path, *, model_dir_exists: bool, resolve_returns: object | None
):
    """Run plan() for a 2-node instance whose model has DownloadCompleted on both nodes.

    ``model_dir_exists`` controls whether ``<tmp_path>/<model>`` exists on disk
    (the guard's ``any_dir_exists`` fast path), and ``resolve_returns`` controls
    what ``resolve_existing_model`` returns (a Path when the model is truly
    complete on disk, None when it is a ghost).
    """
    if model_dir_exists:
        (tmp_path / MODEL_A_ID.normalize()).mkdir(parents=True, exist_ok=True)
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerConnected()
    )

    with (
        patch.object(plan_mod, "EXO_MODELS_DIRS", (tmp_path,)),
        patch.object(plan_mod, "EXO_MODELS_READ_ONLY_DIRS", ()),
        patch.object(plan_mod, "resolve_existing_model", return_value=resolve_returns),
    ):
        return plan_mod.plan(
            node_id=NODE_A,
            runners={RUNNER_1_ID: runner},
            global_download_status={
                NODE_A: [
                    DownloadCompleted(
                        shard_metadata=shard, node_id=NODE_A, total=Memory()
                    )
                ],
                NODE_B: [
                    DownloadCompleted(
                        shard_metadata=shard2, node_id=NODE_B, total=Memory()
                    )
                ],
            },
            instances={INSTANCE_1_ID: instance},
            all_runners={
                RUNNER_1_ID: RunnerConnected(),
                RUNNER_2_ID: RunnerConnected(),
            },
            tasks={},
            input_chunk_buffer={},
            image_cache={},
            instance_backoff=KeyedBackoff(),
            download_backoff=KeyedBackoff(),
        )


def test_ghost_completed_with_no_model_dir_triggers_redownload(tmp_path) -> None:
    """DownloadCompleted with NO model dir anywhere is a ghost -> DownloadModel."""

    result = _plan_for_download(
        tmp_path, model_dir_exists=False, resolve_returns=None
    )
    assert isinstance(result, plan_mod.DownloadModel), (
        "ghost DownloadCompleted (no dir) must trigger a re-download, "
        f"got {type(result).__name__}"
    )


def test_ghost_completed_with_incomplete_dir_triggers_redownload(tmp_path) -> None:
    """DownloadCompleted with a dir that fails resolve_existing_model is a ghost."""

    result = _plan_for_download(
        tmp_path, model_dir_exists=True, resolve_returns=None
    )
    assert isinstance(result, plan_mod.DownloadModel), (
        "ghost DownloadCompleted (incomplete dir) must trigger a re-download, "
        f"got {type(result).__name__}"
    )


def test_completed_with_complete_dir_does_not_redownload(tmp_path) -> None:
    """DownloadCompleted with a genuinely complete dir must NOT re-download."""

    result = _plan_for_download(
        tmp_path,
        model_dir_exists=True,
        resolve_returns=tmp_path / MODEL_A_ID.normalize(),
    )
    assert not isinstance(result, plan_mod.DownloadModel), (
        "a genuinely complete model must be trusted, not re-downloaded"
    )
