"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

from mlipaudit.benchmarks.conformer_selection.conformer_selection import (
    ConformerSelectionBenchmark,
)
from mlipaudit.benchmarks.solvent_radial_distribution.solvent_radial_distribution import (  # noqa: E501
    SolventRadialDistributionBenchmark,
)
from mlipaudit.benchmarks.tautomers.tautomers import TautomersBenchmark


class MlPegConformerSelectionBenchmark(ConformerSelectionBenchmark):
    """
    ConformerSelectionBenchmark wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ASE ``Calculator`` objects
    do not expose ``allowed_atomic_numbers``.
    """

    skip_if_elements_missing = False


class MlPegSolventRadialDistributionBenchmark(SolventRadialDistributionBenchmark):
    """
    ``SolventRadialDistributionBenchmark`` wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ml-peg's ASE ``Calculator``
    objects do not expose the set of elements the underlying model supports, so
    the benchmark cannot decide up front whether to skip. Missing element errors
    are instead handled at runtime.
    """

    skip_if_elements_missing = False


class MlPegTautomersBenchmark(TautomersBenchmark):
    """
    TautomersBenchmark wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ASE ``Calculator`` objects
    do not expose ``allowed_atomic_numbers``.
    """

    skip_if_elements_missing = False
