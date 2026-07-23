"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

from mlipaudit.benchmarks.reference_geometry_stability.reference_geometry_stability import (  # noqa: E501
    ReferenceGeometryStabilityBenchmark,
)


class MlPegReferenceGeometryStabilityBenchmark(ReferenceGeometryStabilityBenchmark):
    """
    ReferenceGeometryStabilityBenchmark wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ASE ``Calculator`` objects
    do not expose ``allowed_atomic_numbers``.
    """

    skip_if_elements_missing = False
