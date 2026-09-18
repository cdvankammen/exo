"""Hardware backend enum.

:class:`Backend` enumerates supported inference backends (MLX Metal, MLX CPU) used by instance placement."""

from enum import Enum


class Backend(str, Enum):
    MlxMetal = "MlxMetal"
    MlxCpu = "MlxCpu"
    MlxCuda = "MlxCuda"
    Vllm = "Vllm"
