"""Typed diatomic curve schema and local data adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import os

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

StrPath = str | os.PathLike[str]
DEFAULT_DFT_REFERENCE_PATH = (
    f"{os.path.dirname(os.path.dirname(__file__))}/data/diatomics-dft.json.gz"
)


def homo_key(formula: str) -> str:
    """
    Collapse a homonuclear pair label such as ``H-H`` to its element key.

    Parameters
    ----------
    formula
        Element or pair label.

    Returns
    -------
    str
        Element key for homonuclear pairs, otherwise the original label.
    """
    element_1, separator, element_2 = formula.partition("-")
    return element_1 if separator and element_1 == element_2 else formula


class DiatomicCurve:
    """
    Store one validated diatomic energy and Cartesian-force curve.

    Parameters
    ----------
    distances
        Sample separations.
    energies
        Sample energies.
    forces
        Cartesian forces for both atoms.
    """

    distances: np.ndarray
    energies: np.ndarray
    forces: np.ndarray

    def __init__(
        self,
        distances: ArrayLike,
        energies: ArrayLike,
        forces: ArrayLike,
    ) -> None:
        """
        Convert curve data to arrays and validate shapes and sample counts.

        Parameters
        ----------
        distances
            Sample separations.
        energies
            Sample energies.
        forces
            Cartesian forces for both atoms.
        """
        self.distances = np.asarray(distances)
        self.energies = np.asarray(energies)
        self.forces = np.asarray(forces)

        for name, values in (
            ("distances", self.distances),
            ("energies", self.energies),
        ):
            if values.ndim != 1:
                raise ValueError(f"{name} must have shape (n,), got {values.shape}")

        n_distances = len(self.distances)
        if (n_energies := len(self.energies)) != n_distances:
            raise ValueError(
                f"distance and energy counts differ: {n_distances} != {n_energies}"
            )

        if (n_forces := len(self.forces)) != n_distances:
            raise ValueError(
                f"distance and force counts differ: {n_distances} != {n_forces}"
            )
        expected_force_shape = (n_distances, 2, 3)
        if self.forces.shape != expected_force_shape:
            raise ValueError(
                "forces must have shape "
                f"{expected_force_shape}, got {self.forces.shape}"
            )


@dataclass
class DiatomicCurves:
    """Store homo- and heteronuclear curves, each with its own distance grid."""

    homo_nuclear: dict[str, DiatomicCurve]
    hetero_nuclear: dict[str, DiatomicCurve] = field(default_factory=dict)


def load_dft_reference_curves(
    functional: str = "PBE",
    ref_path: StrPath | None = None,
) -> DiatomicCurves:
    """
    Load bundled or custom DFT reference curves for one functional.

    Parameters
    ----------
    functional
        Density functional key in the reference payload.
    ref_path
        Optional custom reference path.

    Returns
    -------
    DiatomicCurves
        DFT reference curves.
    """
    reference_path = os.fspath(ref_path or DEFAULT_DFT_REFERENCE_PATH)
    open_function = gzip.open if reference_path.endswith(".gz") else open
    with open_function(reference_path, mode="rt", encoding="utf-8") as file:
        references = json.load(file)[functional]
    return DiatomicCurves(
        homo_nuclear={
            homo_key(formula): DiatomicCurve(
                distances=curve["distances"],
                energies=curve["energies"],
                forces=curve.get("forces", []),
            )
            for formula, curve in references.items()
        },
    )


def load_ml_peg_curves(
    source: pd.DataFrame | StrPath,
    *,
    include_heteronuclear: bool = True,
) -> DiatomicCurves:
    """
    Load ML-PEG CSV samples as x-aligned two-atom energy and force curves.

    Parameters
    ----------
    source
        Dataframe or CSV path with pair, distance, energy, and projected force columns.
    include_heteronuclear
        Whether to include heteronuclear pairs.

    Returns
    -------
    DiatomicCurves
        Converted homo- and heteronuclear curves.
    """
    dataframe = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    required_columns = {"pair", "distance", "energy", "force_parallel"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing ml-peg diatomics columns: {sorted(missing_columns)}")

    homo_nuclear: dict[str, DiatomicCurve] = {}
    hetero_nuclear: dict[str, DiatomicCurve] = {}
    for pair_label, pair_dataframe in dataframe.groupby(
        "pair", sort=False, dropna=False
    ):
        string_pair_label = str(pair_label)
        elements = string_pair_label.split("-")
        if len(elements) != 2 or not all(elements):
            raise ValueError(
                "pair labels must have form 'Element-Element', "
                f"got {string_pair_label!r}"
            )
        element_1, element_2 = elements
        if element_1 != element_2 and not include_heteronuclear:
            continue
        sorted_dataframe = pair_dataframe.sort_values("distance")
        duplicate_rows = sorted_dataframe[
            sorted_dataframe.duplicated("distance", keep=False)
        ]
        if not duplicate_rows.empty:
            duplicate_distances = duplicate_rows["distance"].unique().tolist()
            raise ValueError(
                f"{string_pair_label} has duplicate distance values: "
                f"{duplicate_distances!r}"
            )
        distances = sorted_dataframe["distance"].to_numpy(dtype=float)
        energies = sorted_dataframe["energy"].to_numpy(dtype=float)
        projected_forces = sorted_dataframe["force_parallel"].to_numpy(dtype=float)
        forces = np.zeros((len(distances), 2, 3), dtype=float)
        forces[:, 0, 0] = -projected_forces
        forces[:, 1, 0] = projected_forces
        curve = DiatomicCurve(distances=distances, energies=energies, forces=forces)
        if element_1 == element_2:
            homo_nuclear[element_1] = curve
        else:
            hetero_nuclear[string_pair_label] = curve

    return DiatomicCurves(
        homo_nuclear=homo_nuclear,
        hetero_nuclear=hetero_nuclear,
    )
