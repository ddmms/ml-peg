"""Run calculations for homo- and heteronuclear diatomics benchmark."""

from __future__ import annotations

import itertools
from pathlib import Path

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii, vdw_alvarez
from ase.io import write
import mlipx
from mlipx.abc import NodeWithCalculator
from mlipx.utils import freeze_copy_atoms
import numpy as np
import pandas as pd
import zntrack

from ml_peg.calcs.utils.utils import chdir
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory for calculator outputs
OUT_PATH = Path(__file__).parent / "outputs"

# Minimum fallback distance grid (Å) when covalent/van der Waals data missing
DISTANCE_MIN_FALLBACK = 0.5
DISTANCE_MAX_FALLBACK = 6.0


def _vdw_radius(element: str) -> float:
    """
    Return the van der Waals radius (Å) for an element.

    Parameters
    ----------
    element
        Chemical symbol.

    Returns
    -------
    float
        Van der Waals radius in Ångström, with a conservative fallback.
    """
    z = atomic_numbers[element]
    if z < len(vdw_alvarez.vdw_radii):
        value = vdw_alvarez.vdw_radii[z]
        if not np.isnan(value):
            return float(value)
    return 2.0  # conservative fallback


def _distance_grid(
    element1: str,
    element2: str,
    mode: str,
    min_distance: float,
    max_distance: float,
    n_points: int,
) -> np.ndarray:
    """
    Return distance grid for a diatomic pair.

    Parameters
    ----------
    element1, element2
        Chemical symbols of the atoms forming the diatomic.
    mode
        Distance sampling mode. ``"covalent-radius"`` uses element radii, any
        other value falls back to an evenly spaced grid between ``min_distance``
        and ``max_distance``.
    min_distance, max_distance
        Bounds of the distance grid when ``mode`` is not ``"covalent-radius"``.
    n_points
        Number of points for the uniform grid fallback.

    Returns
    -------
    np.ndarray
        Monotonic array of bond lengths in Ångström.
    """
    if mode == "covalent-radius":
        cov1 = covalent_radii[atomic_numbers[element1]]
        cov2 = covalent_radii[atomic_numbers[element2]]
        rmin = 0.9 * float(cov1 + cov2) / 2.0

        rvdw = 0.5 * (_vdw_radius(element1) + _vdw_radius(element2))
        rmax = max(rmin + 0.5, 3.1 * rvdw)
        rmin = max(DISTANCE_MIN_FALLBACK, rmin)
        step = 0.01
        n_samples = max(int(np.ceil((rmax - rmin) / step)), 2)
        return np.linspace(rmin, rmax, n_samples, dtype=float)

    return np.linspace(min_distance, max_distance, n_points, dtype=float)


class DiatomicsBenchmark(zntrack.Node):
    """
    Benchmark model on homo- and heteronuclear diatomic curves.

    The benchmark samples interaction energies for pairs of species over a bond-length
    grid and stores the resulting trajectories for further analysis.
    """

    model: NodeWithCalculator = zntrack.deps()
    elements: list[str] = zntrack.params(
        ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    )
    include_hetero: bool = zntrack.params(True)
    distance_mode: str = zntrack.params("covalent-radius")
    min_distance: float = zntrack.params(0.8)
    max_distance: float = zntrack.params(6.0)
    n_points: int = zntrack.params(200)

    def run(self) -> None:
        """Execute diatomic calculations for the configured model."""
        calc = self.model.get_calculator()
        write_dir = OUT_PATH / self.model_name
        write_dir.mkdir(parents=True, exist_ok=True)

        pairs = [(el, el) for el in self.elements]
        if self.include_hetero:
            pairs.extend(itertools.combinations(sorted(set(self.elements)), 2))

        records: list[dict[str, float]] = []

        for element1, element2 in pairs:
            pair_label = (
                f"{element1}-{element2}"
                if element1 != element2
                else f"{element1}-{element2}"
            )
            distances = _distance_grid(
                element1,
                element2,
                self.distance_mode,
                self.min_distance,
                self.max_distance,
                self.n_points,
            )

            structures: list[Atoms] = []
            for distance in distances:
                atoms = Atoms(
                    [element1, element2],
                    positions=[(0.0, 0.0, 0.0), (0.0, 0.0, float(distance))],
                )
                atoms.info.setdefault("spin", 1)

                atoms.calc = calc
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces()

                bond_vec = atoms.positions[1] - atoms.positions[0]
                norm = np.linalg.norm(bond_vec)
                direction = bond_vec / norm if norm > 0 else np.zeros(3)
                force_proj = float(np.dot(forces[1], direction))

                atoms_copy = freeze_copy_atoms(atoms)
                atoms_copy.calc = None
                atoms_copy.info.update(
                    {
                        "pair": pair_label,
                        "distance": float(distance),
                        "energy": energy,
                        "force_parallel": force_proj,
                        "model": self.model_name,
                    }
                )
                atoms_copy.arrays["forces"] = forces
                structures.append(atoms_copy)

                records.append(
                    {
                        "pair": pair_label,
                        "element_1": element1,
                        "element_2": element2,
                        "distance": float(distance),
                        "energy": energy,
                        "force_parallel": force_proj,
                    }
                )

            if structures:
                write(write_dir / f"{pair_label}.xyz", structures, format="extxyz")

        if records:
            df = pd.DataFrame.from_records(records)
            df.to_csv(write_dir / "diatomics.csv", index=False)


def build_project(repro: bool = False) -> None:
    """
    Build mlipx project for diatomics benchmark.

    Parameters
    ----------
    repro
        Whether to run ``dvc repro -f`` after building.
    """
    project = mlipx.Project()

    for model_name, model in MODELS.items():
        with project.group(model_name):
            DiatomicsBenchmark(
                model=model,
                model_name=model_name,
            )

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_diatomics() -> None:
    """Run diatomics benchmark via pytest."""
    build_project(repro=True)
