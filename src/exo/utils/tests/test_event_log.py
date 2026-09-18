import errno
from pathlib import Path

import pytest

from exo.shared.types.events import TestEvent
from exo.utils.disk_event_log import DiskEventLog


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    return tmp_path / "event_log"


def test_append_and_read_back(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(5)]
    for e in events:
        log.append(e)

    assert len(log) == 5

    result = list(log.read_all())
    assert len(result) == 5
    for original, restored in zip(events, result, strict=True):
        assert original.event_id == restored.event_id

    log.close()


def test_read_range(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(10)]
    for e in events:
        log.append(e)

    result = list(log.read_range(3, 7))
    assert len(result) == 4
    for i, restored in enumerate(result):
        assert events[3 + i].event_id == restored.event_id

    log.close()


def test_read_range_bounds(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(3)]
    for e in events:
        log.append(e)

    # Start beyond count
    assert list(log.read_range(5, 10)) == []
    # Negative start
    assert list(log.read_range(-1, 2)) == []
    # End beyond count is clamped
    result = list(log.read_range(1, 100))
    assert len(result) == 2

    log.close()


def test_empty_log(log_dir: Path):
    log = DiskEventLog(log_dir)
    assert len(log) == 0
    assert list(log.read_all()) == []
    assert list(log.read_range(0, 10)) == []
    log.close()


def _archives(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("events.*.bin.zst"))


def test_rotation_on_close(log_dir: Path):
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    log.close()

    active = log_dir / "events.bin"
    assert not active.exists()

    archives = _archives(log_dir)
    assert len(archives) == 1
    assert archives[0].stat().st_size > 0


def test_rotation_on_construction_with_stale_file(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "events.bin").write_bytes(b"stale data")

    log = DiskEventLog(log_dir)
    archives = _archives(log_dir)
    assert len(archives) == 1
    assert archives[0].exists()
    assert len(log) == 0

    log.close()


def test_empty_log_no_archive(log_dir: Path):
    """Closing an empty log should not leave an archive."""
    log = DiskEventLog(log_dir)
    log.close()

    active = log_dir / "events.bin"

    assert not active.exists()
    assert _archives(log_dir) == []


def test_close_is_idempotent(log_dir: Path):
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    log.close()
    archive = _archives(log_dir)
    log.close()  # should not raise

    assert _archives(log_dir) == archive


def test_successive_sessions(log_dir: Path):
    """Simulate two master sessions: both archives should be kept."""
    log1 = DiskEventLog(log_dir)
    log1.append(TestEvent())
    log1.close()

    first_archive = _archives(log_dir)[-1]

    log2 = DiskEventLog(log_dir)
    log2.append(TestEvent())
    log2.append(TestEvent())
    log2.close()

    # Session 1 archive shifted to slot 2, session 2 in slot 1
    second_archive = _archives(log_dir)[-1]
    should_be_first_archive = _archives(log_dir)[-2]

    assert first_archive.exists()
    assert second_archive.exists()
    assert first_archive != second_archive
    assert should_be_first_archive == first_archive


def test_rotation_keeps_at_most_5_archives(log_dir: Path):
    """After 7 sessions, only the 5 most recent archives should remain."""
    all_archives: list[Path] = []
    for _ in range(7):
        log = DiskEventLog(log_dir)
        log.append(TestEvent())
        log.close()
        all_archives.append(_archives(log_dir)[-1])

    for old in all_archives[:2]:
        assert not old.exists()
    for recent in all_archives[2:]:
        assert recent.exists()


def test_stale_close_does_not_remove_successor_active_log(log_dir: Path):
    """A stale close must not rotate the active file a successor recreated.

    Regression test for the file-ownership race: an old owner closing after
    a successor process has recreated events.bin at the same path used to
    archive and unlink the successor's active log, breaking event replay.
    """
    stale = DiskEventLog(log_dir)
    stale.append(TestEvent())

    successor = DiskEventLog(log_dir)
    successor_event = TestEvent()
    successor.append(successor_event)

    archives_before = _archives(log_dir)
    stale.close()

    assert (log_dir / "events.bin").exists()
    assert _archives(log_dir) == archives_before
    restored = list(successor.read_all())
    assert len(restored) == 1
    assert restored[0].event_id == successor_event.event_id

    successor.close()


def test_stale_close_of_empty_log_does_not_unlink_successor(log_dir: Path):
    """The empty-log unlink branch must also respect file ownership."""
    stale = DiskEventLog(log_dir)

    successor = DiskEventLog(log_dir)
    successor.append(TestEvent())

    stale.close()

    assert (log_dir / "events.bin").exists()
    assert len(list(successor.read_all())) == 1

    successor.close()


def test_concurrent_append_and_read_are_serialized(log_dir: Path):
    """Thread-safety regression: concurrent append/read on one DiskEventLog
    must not raise (torn frames, OrderedDict-mutated-during-iteration, or
    flush-of-closed-file) and reads must observe contiguous indices.
    """
    import threading

    log = DiskEventLog(log_dir)
    n_writers = 3
    n_readers = 2
    events_per_writer = 250
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer(wid: int) -> None:
        try:
            for _ in range(events_per_writer):
                log.append(TestEvent())
        except BaseException as e:  # pragma: no cover - failure path
            errors.append(e)
        finally:
            stop.set()

    def reader(rid: int) -> None:
        try:
            while not stop.is_set():
                events = list(log.read_all())
                # Every record must deserialize to a complete event and the
                # two read paths must agree on the count (no torn frames).
                assert all(isinstance(e, TestEvent) for e in events)
                ranged = list(log.read_range(0, len(events)))
                assert len(ranged) == len(events), (rid, len(ranged), len(events))
        except BaseException as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [
        *[threading.Thread(target=writer, args=(i,)) for i in range(n_writers)],
        *[threading.Thread(target=reader, args=(i,)) for i in range(n_readers)],
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent access raised: {errors!r}"
    assert len(log) == n_writers * events_per_writer
    final = list(log.read_all())
    assert len(final) == len(log)
    # UUIDs must all be unique (no duplicated/torn records)
    assert len({e.event_id for e in final}) == len(final)

    log.close()


def test_concurrent_close_is_idempotent(log_dir: Path):
    """close() racing threads must not raise or leave a half-rotated log."""
    import threading

    log = DiskEventLog(log_dir)
    log.append(TestEvent())

    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def closer() -> None:
        try:
            barrier.wait()
            log.close()
        except BaseException as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=closer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"concurrent close raised: {errors!r}"
    # active file rotated away exactly once (empty-log unlink suppressed by
    # ownership check after first close)
    assert not (log_dir / "events.bin").exists()
    assert len(_archives(log_dir)) == 1


# --- FIX(t_5b65f607): ENOSPC / unwritable disk must not crash the node ---


def test_append_enospc_drops_event_and_survives(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A full disk (ENOSPC) during append must not raise; the event is dropped
    and subsequent appends still work once the disk is writable again."""
    log = DiskEventLog(log_dir)
    try:
        # Simulate a full disk on the first append.
        def failing_write(data: bytes) -> int:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(
            log._file, "write", failing_write  # pyright: ignore[reportPrivateUsage]
        )
        log.append(TestEvent())  # must not raise

        # Disk "recovered" — the real file handle still works.
        monkeypatch.undo()
        log.append(TestEvent())
        restored = list(log.read_all())
        assert len(restored) == 1
    finally:
        log.close()


def test_append_eacces_erofs_dropped(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """EACCES / EROFS during append are treated like ENOSPC (drop, survive)."""
    for err in (errno.EACCES, errno.EROFS):
        log = DiskEventLog(log_dir)
        try:
            def failing_write(data: bytes, _err: int = err) -> int:
                raise OSError(_err, "unwritable")

            monkeypatch.setattr(
                log._file, "write", failing_write  # pyright: ignore[reportPrivateUsage]
            )
            log.append(TestEvent())  # must not raise
        finally:
            monkeypatch.undo()
            log.append(TestEvent())
            assert len(log) == 1
            log.close()


def test_append_other_oserror_still_raises(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Non-disk OSErrors (e.g. EINVAL) must still propagate — we only swallow
    disk-full / permission-class failures."""
    log = DiskEventLog(log_dir)
    try:
        def failing_write(data: bytes) -> int:
            raise OSError(errno.EINVAL, "Invalid argument")

        monkeypatch.setattr(
            log._file, "write", failing_write  # pyright: ignore[reportPrivateUsage]
        )
        with pytest.raises(OSError):
            log.append(TestEvent())
    finally:
        monkeypatch.undo()
        log.close()


def test_close_enospc_does_not_crash(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """close() during a full disk must not raise (rotation failure is
    swallowed with a warning; the active file is left in place)."""
    log = DiskEventLog(log_dir)
    log.append(TestEvent())

    def failing_rotate(source: object, directory: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(DiskEventLog, "_rotate", staticmethod(failing_rotate))
    try:
        log.close()  # must not raise
    finally:
        monkeypatch.undo()

    # The active file remains (not rotated, not deleted).
    assert (log_dir / "events.bin").exists()
