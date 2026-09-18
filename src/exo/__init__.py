"""exo: run your own AI cluster.

exo is a from-scratch, local-first distributed inference system that turns a cluster of everyday devices into a single AI supercomputer. This package exposes the installed version and hosts the CLI entry point in ``__main__``."""

from importlib.metadata import version

__version__ = version("exo")
