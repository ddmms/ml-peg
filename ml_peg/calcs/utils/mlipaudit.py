"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

import functools
import logging
from typing import Any

from ase import Atoms
from mlip.simulation import SimulationState
from mlipaudit.benchmarks import (
    BondLengthDistributionBenchmark,
    ConformerSelectionBenchmark,
    FoldingStabilityBenchmark,
    InferenceSpeedBenchmark,
    NVEEnergyConservationBenchmark,
    ReactivityBenchmark,
    ReferenceGeometryStabilityBenchmark,
    RingPlanarityBenchmark,
    StabilityBenchmark,
    TautomersBenchmark,
)
from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import (
    NudgedElasticBandBenchmark,
)
from mlipaudit.benchmarks.reference_geometry_stability import (
    reference_geometry_stability as rgs,
)
from mlipaudit.utils.simulation import get_simulation_engine

logger = logging.getLogger(__name__)


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


def run_simulation_saving_final_frame(
    atoms: Atoms, force_field: Any, **kwargs: Any
) -> SimulationState | None:
    """
    Run a simulation through mlip, saving the frame it finished on.

    Replaces ``mlipaudit.utils.run_simulation``. mlip copies recorded frames into the
    simulation state only every ``log_interval`` steps, so a minimization that
    converges before the final step never saves its relaxed geometry.

    Parameters
    ----------
    atoms
        Structure to simulate.
    force_field
        Calculator or force field to run the simulation with.
    **kwargs
        Simulation configuration options.

    Returns
    -------
    SimulationState | None
        Final simulation state, or `None` if the simulation failed.
    """
    engine = get_simulation_engine(atoms=atoms, force_field=force_field, **kwargs)

    try:
        engine.run()
        # Record the frame the simulation finished on, then flush the interval.
        engine._update_temporary_state()
        engine._update_state(engine.state.step, 0.0)
    except Exception as exc:
        logger.info("Error running simulation on system %s: %s", atoms, exc)
        return None

    return engine.state


class MlPegReferenceGeometryStabilityBenchmark(
    MlPegAuditBenchmark, ReferenceGeometryStabilityBenchmark
):
    """``ReferenceGeometryStabilityBenchmark`` wired up for ml-peg."""

    def run_model(self) -> None:
        """Minimize each molecule, saving the geometry it converged to."""
        run_simulation = rgs.run_simulation
        rgs.run_simulation = run_simulation_saving_final_frame
        try:
            super().run_model()
        finally:
            rgs.run_simulation = run_simulation
