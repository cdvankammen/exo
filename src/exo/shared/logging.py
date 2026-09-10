import logging
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol, cast

import zstandard
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

if False:
    # Type-only import: `Record` exists in loguru's .pyi stub but is not a
    # runtime symbol (older loguru versions).
    pass  # pyright: ignore[reportUnusedImport]

_MAX_LOG_ARCHIVES = 5


class _LogLevel(Protocol):
    """Loguru's record['level'] object (has ``.no``)."""

    no: int

# TODO #27: per-module log filters. Loguru handler levels are checked before
# the filter runs, so the handler level is pinned to TRACE and the filter does
# the per-record threshold decision.
LOG_LEVEL_NO: dict[str, int] = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_LOG_LEVELS_ENV = "EXO_LOG_LEVEL"
_MODULE_LEVELS_ENV = "EXO_LOG_LEVELS"


def parse_module_levels(raw: str | None) -> dict[str, str]:
    """Parse the EXO_LOG_LEVELS spec: "module=LEVEL,module=LEVEL".

    Module names are prefixes: "exo.master" also applies to "exo.master.main".
    Invalid entries are skipped so a typo never breaks startup.
    """
    overrides: dict[str, str] = {}
    if not raw:
        return overrides
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        module, _, level = part.partition("=")
        module = module.strip()
        level = level.strip().upper()
        if module and level in LOG_LEVEL_NO:
            overrides[module] = level
    return overrides


def make_module_filter(
    overrides: dict[str, str], base_level: str
) -> Callable[[object], bool]:
    """Return a loguru filter applying per-module levels over a base level.

    The most specific (longest) matching module prefix wins; modules without
    an override use ``base_level``.
    """
    base_no = LOG_LEVEL_NO[base_level.upper()]
    sorted_prefixes = sorted(overrides, key=len, reverse=True)

    def _filter(record: object) -> bool:
        record_dict = cast("dict[str, object]", record)
        name = str(record_dict.get("name") or "")
        threshold = base_no
        for prefix in sorted_prefixes:
            if name == prefix or name.startswith(prefix + "."):
                threshold = LOG_LEVEL_NO[overrides[prefix]]
                break
        level = cast("_LogLevel", record_dict.get("level"))
        return level.no >= threshold

    return _filter


class _ModuleLevelFilter:
    """Picklable per-module log-level filter.

    loguru is configured with ``enqueue=True``, which serializes log
    messages (and the filter) across threads/processes. A local closure
    (``make_module_filter.<locals>._filter``) cannot be pickled and crashed
    runner multiprocessing spawns with:
      AttributeError: Can't get local object 'make_module_filter.<locals>._filter'
    This class is module-level (hence picklable) and applies the same
    most-specific-prefix-wins logic.
    """

    def __init__(self, overrides: dict[str, str], base_level: str):
        self._base_no = LOG_LEVEL_NO[base_level.upper()]
        self._sorted_prefixes = sorted(overrides, key=len, reverse=True)
        self._overrides = dict(overrides)

    def __call__(self, record: dict[str, object]) -> bool:
        name = str(record.get("name") or "")
        threshold = self._base_no
        for prefix in self._sorted_prefixes:
            if name == prefix or name.startswith(prefix + "."):
                threshold = LOG_LEVEL_NO[self._overrides[prefix]]
                break
        level = cast("_LogLevel", record.get("level"))
        return level.no >= threshold


def resolve_base_level(verbosity: int) -> str:
    """Base log level: EXO_LOG_LEVEL env wins, else DEBUG at verbosity > 0."""
    env_level = os.environ.get(_LOG_LEVELS_ENV)
    if env_level and env_level.strip().upper() in LOG_LEVEL_NO:
        return env_level.strip().upper()
    return "DEBUG" if verbosity > 0 else "INFO"


def _zstd_compress(filepath: str) -> None:
    source = Path(filepath)
    dest = source.with_suffix(source.suffix + ".zst")
    cctx = zstandard.ZstdCompressor()
    with open(source, "rb") as f_in, open(dest, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    source.unlink()


def _once_then_never() -> Iterator[bool]:
    yield True
    while True:
        yield False


class InterceptLogger(HypercornLogger):
    def __init__(self, config: Config):
        super().__init__(config)
        assert self.error_logger
        self.error_logger.handlers = [_InterceptHandler()]


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(depth=3, exception=record.exc_info).log(level, record.getMessage())


def logger_setup(log_file: Path | None, verbosity: int = 0):
    """Set up logging for this process - formatting, file handles, verbosity and output"""

    logging.getLogger("exo_rs").setLevel(logging.INFO)
    logging.getLogger("networking").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.remove()

    # replace all stdlib loggers with _InterceptHandlers that log to loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0)

    # TODO #27: per-module log filters. Base level: EXO_LOG_LEVEL env >
    # verbosity flag. EXO_LOG_LEVELS="exo.master=DEBUG,exo.worker=WARNING"
    # overrides individual modules. The handler level is pinned to TRACE and
    # the filter does the threshold decision (loguru checks level before
    # filter).
    base_level = resolve_base_level(verbosity)
    module_levels = parse_module_levels(os.environ.get(_MODULE_LEVELS_ENV))
    log_filter = _ModuleLevelFilter(module_levels, base_level)

    # diagnose=False everywhere: loguru's variable inspection calls repr() on
    # every local in the traceback, and repr of MLX device objects can crash
    # the process mid-log (observed as an opaque SIGSEGV replacing the actual
    # runner exception on the CUDA backend).

    logger.add(
        sys.__stderr__,  # type: ignore
        format=(
            "[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>"
            if verbosity == 0
            else "[ {time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} ] <level>{message}</level>"
        ),
        level=0,  # filter decides
        # mypy/pyright: loguru's FilterFunction is Callable[[Dict[str, Any]], bool];
        # our picklable _ModuleLevelFilter matches structurally.
        filter=log_filter,  # type: ignore[reportArgumentType]
        colorize=True,
        enqueue=True,
        diagnose=False,
    )
    if log_file:
        rotate_once = _once_then_never()
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level=0,  # filter decides
            filter=log_filter,  # type: ignore[reportArgumentType] — picklable filter, see above.
            colorize=False,
            enqueue=True,
            diagnose=False,
            rotation=lambda _, __: next(rotate_once),
            retention=_MAX_LOG_ARCHIVES,
            compression=_zstd_compress,
        )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()


""" --- TODO: Capture MLX Log output:
import contextlib
import sys
from loguru import logger

class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass

logger.remove()
logger.add(sys.__stdout__)

stream = StreamToLogger()
with contextlib.redirect_stdout(stream):
    print("Standard output is sent to added handlers.")
"""
