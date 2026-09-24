"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

import functools

from mlipaudit.benchmarks import (
    BondLengthDistributionBenchmark,
    ConformerSelectionBenchmark,
    FoldingStabilityBenchmark,
    InferenceSpeedBenchmark,
    NVEEnergyConservationBenchmark,
    ReactivityBenchmark,
    RingPlanarityBenchmark,
    StabilityBenchmark,
    TautomersBenchmark,
)
from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import (
    NudgedElasticBandBenchmark,
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


class MlPegFoldingStabilityBenchmark(MlPegAuditBenchmark, FoldingStabilityBenchmark):
    """``FoldingStabilityBenchmark`` wired up for ml-peg."""


class MlPegGrambowBarrierHeightsBenchmark(MlPegAuditBenchmark, ReactivityBenchmark):
    """``ReactivityBenchmark`` wired up for ml-peg."""


class MlPegInferenceSpeedBenchmark(MlPegAuditBenchmark, InferenceSpeedBenchmark):
    """
    ``InferenceSpeedBenchmark`` wired up for ml-peg.

    The MD runs are disabled, as ml-peg only reports forward-pass timings and the
    MD runs would otherwise dominate the benchmark's runtime.
    """

    @functools.cached_property
    def _md_backends(self) -> list[str]:
        """
        Get the MD backends to run.

        Returns
        -------
        list[str]
            No backends, so that only the forward pass is timed.
        """
        return []


class MlPegRingPlanarityBenchmark(MlPegAuditBenchmark, RingPlanarityBenchmark):
    """``RingPlanarityBenchmark`` wired up for ml-peg."""


class MlPegTautomersBenchmark(MlPegAuditBenchmark, TautomersBenchmark):
    """``TautomersBenchmark`` wired up for ml-peg."""


class MlPegStabilityBenchmark(MlPegAuditBenchmark, StabilityBenchmark):
    """``StabilityBenchmark`` wired up for ml-peg."""


class MlPegGrambowOrganicsBenchmark(MlPegAuditBenchmark, NudgedElasticBandBenchmark):
    """``NudgedElasticBandBenchmark`` wired up for ml-peg."""


class MlPegNVEEnergyConservationBenchmark(
    MlPegAuditBenchmark, NVEEnergyConservationBenchmark
):
    """``NVEEnergyConservationBenchmark`` wired up for ml-peg."""
