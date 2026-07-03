"""Adapters for using mlipaudit benchmarks with ml-peg's ASE calculators."""

from __future__ import annotations

from mlipaudit.benchmarks.bond_length_distribution.bond_length_distribution import (
    BondLengthDistributionBenchmark,
)


class MlPegBondLengthDistributionBenchmark(BondLengthDistributionBenchmark):
    """
    ``BondLengthDistributionBenchmark`` wired up for ml-peg's ASE calculators.

    ``skip_if_elements_missing`` is disabled because ml-peg's ASE ``Calculator``
    objects do not expose the set of elements the underlying model supports, so
    the benchmark cannot decide up front whether to skip. Missing element errors
    are instead handled at runtime.
    """

    skip_if_elements_missing = False
