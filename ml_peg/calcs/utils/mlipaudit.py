"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import (
    NudgedElasticBandBenchmark,
)


class MlPegGrambowOrganicsBenchmark(NudgedElasticBandBenchmark):
    """
    NudgedElasticBandBenchmark wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ASE ``Calculator`` objects
    do not expose ``allowed_atomic_numbers``.
    """

    skip_if_elements_missing = False
