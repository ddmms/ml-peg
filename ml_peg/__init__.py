"""Testing framework for machine learnt interatomic potentials."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ml-peg")
except PackageNotFoundError:
    __version__ = "0.0.0"
