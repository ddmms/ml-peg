"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

from mlipaudit.benchmarks.stability.stability import StabilityBenchmark


class MlPegStabilityBenchmark(StabilityBenchmark):
    """
    StabilityBenchmark wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ASE ``Calculator`` objects
    do not expose ``allowed_atomic_numbers``.
    """

    skip_if_elements_missing = False
