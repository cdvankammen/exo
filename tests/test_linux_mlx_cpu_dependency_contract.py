import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CUDA_EXTRAS = {"mlx-cuda12", "mlx-cuda13"}


def _extras_a_source_applies_to(source: dict) -> set[str]:
    """Extras a [tool.uv.sources] entry can be selected for.

    uv offers two equivalent ways to scope a source to an extra: the dedicated
    `extra = "..."` key, or an `extra == '...'` clause inside the PEP 508
    marker. Upstream exo #2385 uses the marker form; this fork (4d7a73ca8)
    uses the `extra` key. Read both so the contract survives either spelling.
    """
    if "extra" in source:
        return {source["extra"]}
    return set(re.findall(r"extra\s*==\s*'([^']+)'", source.get("marker", "")))


def test_linux_mlx_cpu_extra_uses_a_matching_registry_pair() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    optional = pyproject["project"]["optional-dependencies"]

    mlx_requirement = next(
        requirement for requirement in optional["mlx"] if requirement.startswith("mlx==")
    )
    cpu_requirement = next(
        requirement
        for requirement in optional["mlx-cpu"]
        if requirement.startswith("mlx-cpu==")
    )
    mlx_version = mlx_requirement.removeprefix("mlx==")
    cpu_version = cpu_requirement.removeprefix("mlx-cpu==").split(";", 1)[0]

    # The ABI half of #2359/#2382: `mlx` (bindings) and `mlx-cpu` (runtime) must
    # be the same release, or `import mlx.core` dies with `undefined symbol`.
    assert cpu_version == mlx_version

    linux_sources = [
        source
        for source in pyproject["tool"]["uv"]["sources"]["mlx"]
        if "sys_platform == 'linux'" in source.get("marker", "")
    ]
    assert linux_sources, "no Linux mlx sources to gate"

    # The fork's own CUDA `libmlx.so` does not export the symbols the PyPI
    # mlx-cpu runtime links against, so the fork wheels must never be selected
    # for the mlx-cpu extra -- only for the CUDA extras that ship a matching
    # runtime.
    for source in linux_sources:
        extras = _extras_a_source_applies_to(source)
        assert "mlx-cpu" not in extras
        assert extras, f"ungated Linux mlx source: {source}"
        assert extras <= CUDA_EXTRAS

    # Every CUDA extra must still have at least one fork wheel available.
    covered = set().union(*(_extras_a_source_applies_to(s) for s in linux_sources))
    assert covered == CUDA_EXTRAS


if __name__ == "__main__":
    test_linux_mlx_cpu_extra_uses_a_matching_registry_pair()