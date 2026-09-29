"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

from unittest.mock import patch

from mlipaudit.benchmarks import (
    BondLengthDistributionBenchmark,
    ConformerSelectionBenchmark,
    NVEEnergyConservationBenchmark,
    TautomersBenchmark,
)
from mlipaudit.benchmarks.nve_energy_conservation import nve_energy_conservation


class MlPegBondLengthDistributionBenchmark(BondLengthDistributionBenchmark):
    """
    ``BondLengthDistributionBenchmark`` wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ml-peg's ASE ``Calculator``
    objects do not expose the set of elements the underlying model supports, so
    the benchmark cannot decide up front whether to skip. Missing element errors
    are instead handled at runtime.
    """

    skip_if_elements_missing = False


class MlPegConformerSelectionBenchmark(ConformerSelectionBenchmark):
    """
    ConformerSelectionBenchmark wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ASE ``Calculator`` objects
    do not expose ``allowed_atomic_numbers``.
    """

    skip_if_elements_missing = False


class MlPegTautomersBenchmark(TautomersBenchmark):
    """
    TautomersBenchmark wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ASE ``Calculator`` objects
    do not expose ``allowed_atomic_numbers``.
    """

    skip_if_elements_missing = False


class MlPegNVEEnergyConservationBenchmark(NVEEnergyConservationBenchmark):
    """
    ``NVEEnergyConservationBenchmark`` wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ASE ``Calculator`` objects
    do not expose ``allowed_atomic_numbers``. Additionally, the ``run_model``
    method erroneously calls ``skip_unallowed_elements`` either way, which throws
    an error. We patch the ``run_model`` method to skip this.
    """

    skip_if_elements_missing = False

    def run_model(self) -> None:
        """Run the benchmark without the per-system element pre-check."""
        with patch.object(
            nve_energy_conservation,
            "skip_unallowed_elements",
            lambda force_field, structure_tuples: [],
        ):
            super().run_model()
