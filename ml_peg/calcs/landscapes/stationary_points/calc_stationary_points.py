"""Evaluate MLIPs on the Landscape17 DFT landscapes.

Landscape17 maps the potential energy landscape of six molecules as a network
of minima joined by transition states, computed at wB97X/6-31G(d). It holds 28
minima, 67 transition states, a DFT Hessian for every one of them, and the
steepest-descent paths joining them -- 11,173 frames carrying DFT energies and
forces.

This benchmark hands a model those geometries and asks how much of the
landscape survives contact with it. Nothing here searches for new structures;
the companion ``landscape_search`` benchmark asks whether a model can find the
landscape unaided.

Two kinds of geometry, which test different things:

``test_paths``
    Energies and forces along the steepest-descent paths. These are ordinary
    points on the surface with non-zero forces, so this is a conventional
    accuracy test -- but sampled along the reaction coordinates rather than
    from a thermal distribution, which is where a landscape benchmark should
    want to look.

``test_stationary_points``
    The model's force at each reference minimum and transition state. DFT
    forces there are zero by construction, so what is recorded is the residual:
    how far from stationary the DFT geometry is under the model. This predicts
    whether the structure will survive reconvergence, and is reported rather
    than compared against a reference of zeros.

Still to come:

``test_reconverge_minima``
    Relax each reference minimum under the model. A good model leaves it where
    it is; the distance it drifts is the metric.

``test_reconverge_transition_states``
    The same for saddles, by eigenvector following. Harder, because a saddle is
    only stationary in one direction, and a model may have no saddle nearby at
    all.

``test_connectivity``
    Follow the steepest-descent paths from each reconverged saddle and check
    which minima they reach. A model can place a saddle correctly and still
    connect it to the wrong basins, which changes the kinetics while leaving
    every structure looking right.

``test_hessians``
    Compare the model's Hessian at each reference geometry with the DFT one:
    the overlap of the normal modes, the overlap of the reaction coordinate at
    saddles, and whether the curvature is systematically scaled.

Reported distributions rather than pass rates
---------------------------------------------
The drift and overlap tests report positions in a distribution -- medians and
upper percentiles -- rather than the fraction falling below a threshold. A
fraction is brittle in a way that misleads: a model that reproduces every
structure with the same small systematic displacement scores zero if that
displacement sits a hair outside the cut, which reads as total failure rather
than as the tidy, correctable offset it is. Percentiles say where the
distribution sits and whether it has a tail, and both can be given
``good``/``bad`` thresholds with a physical justification.

Results are kept per molecule rather than pooled. Aspirin alone is 37 of the 67
transition states and 6,944 of the 11,173 path frames, so a single pooled
number is largely a statement about aspirin. Raw outputs are written per
structure, so pooling later is a change to the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from warnings import warn

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
import pytest
from tqdm.auto import tqdm

from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

pytest.importorskip("topsearch", reason="Please install the `topsearch` extra")

from topsearch.data.kinetic_transition_network import (  # noqa: E402
    KineticTransitionNetwork,
)

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"
DATA_URI = (
    "https://github.com/VladCarare/mlp-landscapes/releases/download/landscape17-data-v1"
)
DATA_FILENAME = "Landscape17.zip"

# Every Landscape17 molecule is a neutral closed-shell singlet.
CHARGE = 0
SPIN = 1


@dataclass(frozen=True)
class _Molecule:
    """One Landscape17 molecule.

    Attributes
    ----------
    key
        Name used for outputs and metrics.
    directory
        Directory name inside the archive.
    xyz
        Basename of the reference structure, which supplies the element order.
        It does not always match the directory name.
    """

    key: str
    directory: str
    xyz: str


MOLECULES: tuple[_Molecule, ...] = (
    _Molecule("ethanol", "ethanol", "ethanol.xyz"),
    _Molecule("malonaldehyde", "malonaldehyde", "malonaldehyde.xyz"),
    _Molecule("salicylic_acid", "salicylic_acid", "salicylic.xyz"),
    _Molecule("paracetamol", "paracetamol", "paracetamol.xyz"),
    _Molecule("azobenzene", "azobenzene", "azobenzene.xyz"),
    _Molecule("aspirin", "aspirin", "aspirin.xyz"),
)


def _data_dir() -> Path:
    """
    Fetch and extract the Landscape17 archive.

    Returns
    -------
    pathlib.Path
        Directory holding the six molecule directories.
    """
    return download_github_data(filename=DATA_FILENAME, github_uri=DATA_URI) / (
        "Landscape17"
    )


def _stationary_points(data_dir: Path, molecule: _Molecule) -> list[Atoms]:
    """
    Read a molecule's minima and transition states from the reference network.

    Parameters
    ----------
    data_dir
        Directory holding the extracted archive.
    molecule
        Molecule to read.

    Returns
    -------
    list[ase.atoms.Atoms]
        Minima followed by transition states, each carrying its reference
        energy and its role in ``atoms.info``.
    """
    path = data_dir / molecule.directory
    symbols = read(path / molecule.xyz).get_chemical_symbols()

    network = KineticTransitionNetwork()
    network.read_network(text_path=f"{path}/", text_string="")

    frames: list[Atoms] = []
    for index in list(network.G.nodes):
        atoms = Atoms(symbols, network.get_minimum_coords(index).reshape(-1, 3))
        atoms.info.update(
            molecule=molecule.key,
            kind="minimum",
            label=f"min{index:02d}",
            ref_energy_ev=float(network.get_minimum_energy(index)),
        )
        frames.append(atoms)

    for plus, minus, edge in list(network.G.edges):
        atoms = Atoms(symbols, network.get_ts_coords(plus, minus, edge).reshape(-1, 3))
        atoms.info.update(
            molecule=molecule.key,
            kind="transition_state",
            label=f"ts{plus}_{minus}_{edge}",
            connects=f"{plus},{minus}",
            ref_energy_ev=float(network.get_ts_energy(plus, minus, edge)),
        )
        frames.append(atoms)

    return frames


def _path_frames(data_dir: Path, molecule: _Molecule) -> list[Atoms]:
    """
    Read the steepest-descent path frames, which carry DFT forces.

    Parameters
    ----------
    data_dir
        Directory holding the extracted archive.
    molecule
        Molecule to read.

    Returns
    -------
    list[ase.atoms.Atoms]
        Frames with ``ref_energy_ev`` and ``ref_forces`` set.
    """
    frames: list[Atoms] = []
    root = data_dir / molecule.directory / "SteepestDescent"
    for path_file in sorted(root.glob("*/*.xyz")):
        atoms = read(path_file)
        # extxyz forces and energies are read into a SinglePointCalculator,
        # not into atoms.arrays and atoms.info. Move them across explicitly,
        # then drop the calculator so a model calculator can be attached.
        if atoms.calc is None or "forces" not in atoms.calc.results:
            warn(f"No DFT forces in {path_file}, skipping", stacklevel=2)
            continue
        reference_forces = np.asarray(atoms.calc.results["forces"], dtype=float)
        reference_energy = float(atoms.calc.results["energy"])
        atoms.calc = None
        atoms.info.update(
            molecule=molecule.key,
            kind="path",
            label=f"{path_file.parent.name}/{path_file.stem}",
            ref_energy_ev=reference_energy,
        )
        atoms.arrays["ref_forces"] = reference_forces
        frames.append(atoms)
    return frames


def _evaluate(calc, frames: list[Atoms], description: str) -> list[Atoms]:
    """
    Run single points over a set of frames, recording failures as NaN.

    Parameters
    ----------
    calc
        ASE calculator to evaluate with.
    frames
        Structures to evaluate.
    description
        Label for the progress bar.

    Returns
    -------
    list[ase.atoms.Atoms]
        Copies of the inputs with ``mlip_energy_ev`` and ``mlip_forces`` set.
    """
    iterator = frames
    if sys.stderr.isatty():
        iterator = tqdm(frames, desc=description, unit="pt", leave=False)

    results: list[Atoms] = []
    for atoms in iterator:
        predicted = atoms.copy()
        predicted.calc = calc
        predicted.info["charge"] = CHARGE
        predicted.info["spin"] = SPIN

        out_atoms = atoms.copy()
        try:
            out_atoms.info["mlip_energy_ev"] = float(predicted.get_potential_energy())
            out_atoms.arrays["mlip_forces"] = np.asarray(
                predicted.get_forces(), dtype=float
            )
        except Exception as exc:
            warn(
                f"{description}: failed on {atoms.info['label']}: {exc}",
                stacklevel=2,
            )
            out_atoms.info["mlip_energy_ev"] = np.nan
            out_atoms.arrays["mlip_forces"] = np.full((len(atoms), 3), np.nan)
        results.append(out_atoms)
    return results


@pytest.mark.parametrize("mlip", MODELS.items())
def test_stationary_points(mlip: tuple[str, Any]) -> None:
    """
    Evaluate MLIP energies and residual forces at the DFT stationary points.

    DFT forces are zero at these geometries by construction, so the model's own
    force magnitude is the quantity of interest: it says how far the DFT
    structure is from being stationary under the model, which is what decides
    whether it will survive reconvergence.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    data_dir = _data_dir()
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for molecule in MOLECULES:
        frames = _stationary_points(data_dir, molecule)
        results = _evaluate(calc, frames, f"{model_name} {molecule.key} points")
        write(out_dir / f"{molecule.key}_stationary_points.extxyz", results)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_paths(mlip: tuple[str, Any]) -> None:
    """
    Evaluate MLIP energies and forces along the DFT steepest-descent paths.

    11,173 frames with non-zero DFT forces, sampled along the reaction
    coordinates rather than from a thermal distribution.

    The frame count looks large but the cost is not: a single point on the
    largest molecule takes 17 ms with GFN2-xTB, so the whole set is about three
    minutes. Not marked slow on that basis. A representative GPU model has not
    been timed yet, and the marker should be added if one turns out to be
    expensive.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    data_dir = _data_dir()
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for molecule in MOLECULES:
        frames = _path_frames(data_dir, molecule)
        results = _evaluate(calc, frames, f"{model_name} {molecule.key} paths")
        write(out_dir / f"{molecule.key}_paths.extxyz", results)
