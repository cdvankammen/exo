import asyncio
import os

import pytest
from _pytest.capture import CaptureFixture
from exo_rs import (
    NetworkingHandle,
    Pidfile,
    FromSwarm,
)


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    # Use high ephemeral-ish ports so the test never collides with a running
    # cluster (which occupies 52413/52414). Port 0 is not supported by the
    # Rust binding and now raises ValueError (formerly panicked), so pick a
    # fixed high range.
    h = NetworkingHandle.new(
        os.urandom(16).hex().lstrip("0"), "test", 53414, 53413
    )
    print("PYTHON: handle started")

    rt = asyncio.create_task(_await_recv(h))

    # sleep for 4 ticks
    for i in range(10):
        await asyncio.sleep(1)

        await h.gossipsub_publish("topic", b"somehting or other")


def test_networking_handle_port_zero_raises_value_error() -> None:
    # Regression guard (F-3): listen_port=0 used to hit a todo!() panic
    # (PanicException / abort) in the Rust binding. It must now surface as
    # a normal Python ValueError so callers can handle it.
    with pytest.raises(ValueError, match="cannot listen on port 0 yet"):
        NetworkingHandle.new(os.urandom(16).hex().lstrip("0"), "test", 0, 0)
    print("PYTHON: port-0 rejected with ValueError (no panic)")


def test_pidfile_after_close_raises_runtime_error() -> None:
    # Regression guard (F-2): PyPidfile::get/get_mut used .expect() which
    # surfaced as PanicException after close(). Every method must now raise
    # a catchable RuntimeError with the same message instead of panicking.
    p = Pidfile("/tmp/exo_rs_test_after_close.pid", 0o0600)
    p.write()  # resource is live: write must succeed
    p.close()
    with pytest.raises(RuntimeError, match="cannot use resource after exiting context"):
        p.write()
    with pytest.raises(RuntimeError, match="cannot use resource after exiting context"):
        p.as_raw_fd()
    with pytest.raises(RuntimeError, match="cannot use resource after exiting context"):
        p.close()
    print("PYTHON: after-close use raises RuntimeError (no panic)")


def test_pidfile(capsys: CaptureFixture[str]):
    with capsys.disabled():
        print("\nbefore python")
        scoped_lock_file()
        print("after python")


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case FromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case FromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")


def scoped_lock_file():
    a = Pidfile("/tmp/lock.pid", 0o0600)


if __name__ == "__main__":
    asyncio.run(test_sleep_on_multiple_items())
