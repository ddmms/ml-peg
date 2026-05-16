"""Testing framework for machine learnt interatomic potentials."""

from __future__ import annotations

from importlib.metadata import version

try:
    __version__ = version("ml-peg")
except Exception:
    __version__ = "unknown"
