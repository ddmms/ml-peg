"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

import functools
from typing import Any
from unittest.mock import patch
from warnings import warn

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
from mlipaudit.benchmarks.nve_energy_conservation import nve_energy_conservation
from mlipaudit.utils import run_simulation


def _run_simulation_or_none(*args: Any, **kwargs: Any) -> Any:
    """
    Run an mlipaudit simulation, returning ``None`` if it raises.

    ``None`` is what mlipaudit's NVE ``run_model`` already expects from a
    simulation that produced no usable state, so the failing system is recorded
    as failed and the remaining systems still run.

    Parameters
    ----------
    *args
        Positional arguments for ``run_simulation``.
    **kwargs
        Keyword arguments for ``run_simulation``.

    Returns
    -------
    Any
        Final simulation state, or ``None`` if the simulation raised.
    """
    try:
        return run_simulation(*args, **kwargs)
    except Exception as exc:
        warn(f"Simulation failed: {exc}", stacklevel=2)
        return None


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
    """
    ``NVEEnergyConservationBenchmark`` wired up for ml-peg.

    Two upstream behaviours are patched out:

    ``run_model`` calls ``skip_unallowed_elements`` regardless of
    ``skip_if_elements_missing``, which raises for ASE ``Calculator`` objects.

    ``run_model`` also calls ``run_simulation`` without catching exceptions, so one
    failing system aborts the whole benchmark, losing the systems already
    simulated. Upstream already handles a simulation that returns no state, so
    failures are routed into that path instead.
    """

    def run_model(self) -> None:
        """Run the benchmark, tolerating per-system simulation failures."""
        with (
            patch.object(
                nve_energy_conservation,
                "skip_unallowed_elements",
                lambda force_field, structure_tuples: [],
            ),
            patch.object(
                nve_energy_conservation,
                "run_simulation",
                _run_simulation_or_none,
            ),
        ):
            super().run_model()
