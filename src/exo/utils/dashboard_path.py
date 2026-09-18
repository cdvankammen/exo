import sys
from pathlib import Path
from typing import cast


def find_resources() -> Path:
    resources = _find_in_repo("resources", require_index=False) or _find_in_bundle(
        "resources"
    )
    if resources is None:
        raise FileNotFoundError(
            "Unable to locate resources. Did you clone the repo properly?"
        )
    return resources


def _find_in_repo(subdir: str, require_index: bool) -> Path | None:
    current_module = Path(__file__).resolve()
    for parent in current_module.parents:
        build = parent / subdir
        if build.is_dir() and (not require_index or (build / "index.html").exists()):
            return build
    return None


def _find_in_bundle(subdir: str) -> Path | None:
    frozen_root = cast(str | None, getattr(sys, "_MEIPASS", None))
    if frozen_root is None:
        return None
    candidate = Path(frozen_root) / subdir
    if candidate.is_dir():
        return candidate
    return None


def find_dashboard() -> Path:
    dashboard = _find_in_bundle("dashboard") or _find_in_repo(
        "dashboard/build", require_index=True
    )
    if not dashboard:
        raise FileNotFoundError(
            "Unable to locate dashboard assets - you probably forgot to run `cd dashboard && npm install && npm run build && cd ..`"
        )
    return dashboard
