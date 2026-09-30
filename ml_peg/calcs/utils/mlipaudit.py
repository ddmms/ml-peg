"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

from mlipaudit.benchmarks import (
    BondLengthDistributionBenchmark,
    ConformerSelectionBenchmark,
    ReactivityBenchmark,
    RingPlanarityBenchmark,
    TautomersBenchmark,
)


class MlPegAuditBenchmark:
    """
    Mixin wiring up mlipaudit benchmarks for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ml-peg's ASE ``Calculator``
    objects do not expose the set of elements the underlying model supports, so
    the benchmark cannot decide up front whether to skip. Missing element errors
    are instead handled at runtime.
    """

    skip_if_elements_missing = False


class MlPegBondLengthDistributionBenchmark(
    MlPegAuditBenchmark, BondLengthDistributionBenchmark
):
    """``BondLengthDistributionBenchmark`` wired up for ml-peg."""


class MlPegConformerSelectionBenchmark(
    MlPegAuditBenchmark, ConformerSelectionBenchmark
):
    """``ConformerSelectionBenchmark`` wired up for ml-peg."""


class MlPegGrambowBarrierHeightsBenchmark(MlPegAuditBenchmark, ReactivityBenchmark):
    """``ReactivityBenchmark`` wired up for ml-peg."""


class MlPegRingPlanarityBenchmark(MlPegAuditBenchmark, RingPlanarityBenchmark):
    """``RingPlanarityBenchmark`` wired up for ml-peg's ASE calculators."""


class MlPegTautomersBenchmark(MlPegAuditBenchmark, TautomersBenchmark):
    """``TautomersBenchmark`` wired up for ml-peg."""
