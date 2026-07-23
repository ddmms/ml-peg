"""Typed diatomic curve schema and local data adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import os
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

StrPath = str | os.PathLike[str]
DEFAULT_DFT_REFERENCE_PATH = (
    f"{os.path.dirname(os.path.dirname(__file__))}/data/diatomics-dft.json.gz"
)


def homo_key(formula: str) -> str:
    """Collapse a homonuclear pair label such as ``H-H`` to its element key."""
    element_1, separator, element_2 = formula.partition("-")
    return element_1 if separator and element_1 == element_2 else formula


class DiatomicCurve:
    """Store one validated diatomic energy and Cartesian-force curve."""

    distances: np.ndarray
    energies: np.ndarray
    forces: np.ndarray

    def __init__(
        self,
        distances: ArrayLike,
        energies: ArrayLike,
        forces: ArrayLike,
    ) -> None:
        """Convert curve data to arrays and validate shapes and sample counts."""
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

        # Handle forces stored as (1, n_distances*n_atoms, 3)
        # instead of (n_distances, n_atoms, 3)
        if self.forces.shape == (1, 2 * n_distances, 3):
            self.forces = self.forces.reshape(n_distances, 2, 3)
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
    """Store homo- and heteronuclear curves with their shared or union grid."""

    distances: np.ndarray
    homo_nuclear: dict[str, DiatomicCurve]
    hetero_nuclear: dict[str, DiatomicCurve] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiatomicCurves:
        """Parse MBD JSON curves, requiring per-curve grids to be ordered subsets."""
        distances = np.asarray(data["distances"])
        grid_position_by_distance = {
            float(distance): index for index, distance in enumerate(distances)
        }

        def make_curves(section: str) -> dict[str, DiatomicCurve]:
            """Convert one MBD JSON section to typed curves."""
            raw_curves = data.get(section, data.get(section.replace("-", "_"), {}))
            key_function = homo_key if section.startswith("homo") else str

            def curve_distances(
                formula: str,
                curve: dict[str, Any],
            ) -> np.ndarray:
                """Return an ordered per-curve subset of the top-level grid."""
                curve_distance_array = np.asarray(curve.get("distances", distances))
                # off-grid points map to -1; valid subsets have strictly
                # increasing grid positions
                grid_positions = np.array(
                    [
                        grid_position_by_distance.get(float(distance), -1)
                        for distance in curve_distance_array
                    ]
                )
                if (grid_positions < 0).any() or (np.diff(grid_positions) <= 0).any():
                    raise ValueError(
                        f"{formula} curve distances must be an ordered subset "
                        "of top-level distances"
                    )
                return curve_distance_array

            return {
                key_function(formula): DiatomicCurve(
                    distances=curve_distances(formula, curve),
                    energies=curve["energies"],
                    forces=curve.get("forces", []),
                )
                for formula, curve in raw_curves.items()
                if len(curve["energies"]) > 0
            }

        return cls(
            distances=distances,
            homo_nuclear=make_curves("homo-nuclear"),
            hetero_nuclear=make_curves("hetero-nuclear"),
        )


def _load_json(path: StrPath) -> dict[str, Any]:
    """Load a JSON or gzipped JSON object."""
    string_path = os.fspath(path)
    open_function = gzip.open if string_path.endswith(".gz") else open
    with open_function(string_path, mode="rt", encoding="utf-8") as file:
        return json.load(file)


def load_mbd_json(path: StrPath) -> DiatomicCurves:
    """Load MBD-format predicted curves from JSON or gzipped JSON."""
    return DiatomicCurves.from_dict(_load_json(path))


def load_dft_reference_curves(
    functional: str = "PBE",
    ref_path: StrPath | None = None,
) -> DiatomicCurves:
    """Load bundled or custom DFT reference curves for one functional."""
    reference_path = ref_path or DEFAULT_DFT_REFERENCE_PATH
    references = _load_json(reference_path)[functional]
    return DiatomicCurves(
        distances=np.array([]),
        homo_nuclear={
            homo_key(formula): DiatomicCurve(
                distances=curve["distances"],
                energies=curve["energies"],
                forces=curve.get("forces", []),
            )
            for formula, curve in references.items()
        },
    )


def _parse_pair_label(pair_label: str) -> tuple[str, str]:
    """Parse an ``Element-Element`` pair label."""
    elements = pair_label.split("-")
    if len(elements) != 2 or not all(elements):
        raise ValueError(
            f"pair labels must have form 'Element-Element', got {pair_label!r}"
        )
    return elements[0], elements[1]


def curves_from_ml_peg_dataframe(
    dataframe: pd.DataFrame,
    *,
    include_heteronuclear: bool = True,
) -> DiatomicCurves:
    """Convert an ml-peg dataframe to x-aligned two-atom force curves."""
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
        element_1, element_2 = _parse_pair_label(string_pair_label)
        if element_1 != element_2 and not include_heteronuclear:
            continue
        sorted_dataframe = pair_dataframe.sort_values("distance")
        duplicate_rows = sorted_dataframe[
            sorted_dataframe.duplicated("distance", keep=False)
        ]
        for distance, duplicate_samples in duplicate_rows.groupby("distance"):
            unique_values = duplicate_samples[
                ["energy", "force_parallel"]
            ].drop_duplicates()
            if len(unique_values) > 1:
                raise ValueError(
                    f"{string_pair_label} has conflicting samples at "
                    f"distance={distance}"
                )
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

    included_curves = [*homo_nuclear.values(), *hetero_nuclear.values()]
    all_distances = (
        np.concatenate([curve.distances for curve in included_curves])
        if included_curves
        else np.array([], dtype=float)
    )
    return DiatomicCurves(
        distances=np.sort(np.unique(all_distances)),
        homo_nuclear=homo_nuclear,
        hetero_nuclear=hetero_nuclear,
    )


def load_ml_peg_curves(
    source: pd.DataFrame | StrPath,
    *,
    include_heteronuclear: bool = True,
) -> DiatomicCurves:
    """Load current ml-peg diatomic curves from a dataframe or CSV."""
    dataframe = (
        source if isinstance(source, pd.DataFrame) else pd.read_csv(os.fspath(source))
    )
    return curves_from_ml_peg_dataframe(
        dataframe, include_heteronuclear=include_heteronuclear
    )
