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

``test_reconverge_minima``
    Relax each reference minimum under the model. A good model leaves it where
    it is; the distance it drifts is the metric.

``test_reconverge_transition_states``
    The same for saddles, by eigenvector following, plus the connectivity.
    Harder than the minima, because a saddle is stationary in only one
    direction and a model may have no saddle nearby at all. Connectivity lives
    here rather than in a test of its own because eigenvector following already
    returns the two minima its steepest-descent paths reach, so splitting them
    would mean running the whole search twice for a result already in hand.

Everything that searches runs the model in float64, via
``get_calculator(precision="high")``. That is a requirement rather than a
preference: eigenvector following takes its curvature from finite differences
of forces, which in float32 is noise-dominated, and the search wanders.

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
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

pytest.importorskip("topsearch", reason="Please install the `topsearch` extra")

from topsearch.data.coordinates import MolecularCoordinates  # noqa: E402
from topsearch.data.kinetic_transition_network import (  # noqa: E402
    KineticTransitionNetwork,
)
from topsearch.minimisation import lbfgs  # noqa: E402
from topsearch.potentials.potential import Potential  # noqa: E402
from topsearch.similarity.molecular_similarity import (  # noqa: E402
    MolecularSimilarity,
)
from topsearch.transition_states.hybrid_eigenvector_following import (  # noqa: E402
    HybridEigenvectorFollowing,
)
from topsearch.utils.random import set_global_seed  # noqa: E402

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


# Reconvergence settings. These are the search's parameters, not the model's,
# and every number reported below depends on them, so they are stated here
# rather than left to defaults.
SEED = 0
# pgtol on the largest gradient component, so this is about a thousand times
# tighter than an ordinary geometry optimisation, and it has to be. The metric
# is how far a structure travels, and a loose criterion stops it before it has
# finished travelling, which flatters the model: measured on MACE-OFF, relaxing
# this to 1e-3 cuts paracetamol's median drift from 0.065 to 0.020 A, because
# the last third of the journey happens along soft torsions at gradients below
# 1e-3. The cliff is molecule-dependent -- paracetamol survives 1e-4 and breaks
# at 1e-3, salicylic acid already breaks at 1e-4 -- so there is no looser value
# that is safe across the set. 1e-5 is safe on both but saves only 6 to 25% of
# the iterations, which is not worth the risk on the four molecules not tested.
MINIMISE_CONV_CRIT = 1e-6
# topsearch defaults this to 200, which is not enough: on MACE-OFF these minima
# need 460 to 670 iterations to reach the criterion above, and at 200 every one
# of them stops on the iteration limit instead -- reporting a converged minimum
# from an optimisation that never converged. Raising it to 5000 changes the
# drift by nothing, so 1000 is the budget rather than the ceiling.
MINIMISE_STEPS = 1000
# A fixed protocol rather than a safe cutoff. Measured on MACE-OFF across the
# full reference, the step count does NOT separate outcomes: a saddle can take
# 541 steps and land 0.26 A from the DFT one, or 22 steps and land 0.30 A away.
# So no budget can be justified by a gap in the distribution -- an earlier
# version of this comment claimed one, and aspirin, which is 37 of the 67
# saddles, falsified it.
#
# What the budget does instead is truncate the slow tail, and that tail is
# overwhelmingly made of failures. Measured over all 37 aspirin saddles, which
# are more than half the reference: 18 converge within 150 steps, 13 take
# longer, and of those 13 exactly one lands close to the DFT saddle (0.258 A
# after 541 steps) while the rest end 0.5 to 1.4 A away. Six never converge at
# all, even given 3000 steps. So full completeness is not reachable at any
# sane budget, and 150 costs about one reproduced saddle in 37 while running
# roughly twenty times faster.
#
# Two consequences to state wherever these numbers are quoted. Saddles that
# would have converged slowly are reported as not converged, so this measures
# the model under a fixed search budget rather than in the limit. And because
# the truncated tail is the far-drifting part, the mean drift is pulled down
# towards the median -- on aspirin, mean 0.436 over everything that converged
# against 0.099 inside the budget -- which suppresses the skew that reporting
# both was meant to reveal. `steps` is recorded per saddle so any of this can
# be checked from the output.
TS_STEPS = 150
TS_CONV_CRIT = 1e-2
PUSHOFF = 0.8
MAX_UPHILL_STEP_SIZE = 0.01
POSITIVE_EIGENVALUE_STEP = 0.1
STEEPEST_DESCENT_CONV_CRIT = 1e-3
EIGENVALUE_CONV_CRIT = 5e-2
# Below this RMSD two structures are treated as the same one.
SAME_STRUCTURE = 0.3


class _AseCalculatorPotential(Potential):
    """
    A topsearch potential backed by any ASE calculator.

    topsearch ships its own model loaders, but ml-peg has a model registry
    already and it is the one under test. The search algorithms need only
    three methods, so this adapter lets topsearch supply the algorithms while
    ml-peg supplies the model.

    Parameters
    ----------
    symbols
        Chemical symbols, fixing the element order for every position passed in.
    calc
        ASE calculator to evaluate with.
    """

    def __init__(self, symbols: list[str], calc) -> None:
        super().__init__()
        self.atomistic = True
        self.atoms = Atoms(symbols, positions=np.zeros((len(symbols), 3)))
        self.atoms.info.update(charge=CHARGE, spin=SPIN)
        self.atoms.calc = calc

    def function(self, position: np.ndarray) -> float:
        """
        Evaluate the energy at a position.

        Parameters
        ----------
        position
            Flattened coordinates.

        Returns
        -------
        float
            Potential energy in eV.
        """
        self.atoms.set_positions(position.reshape(-1, 3))
        return float(self.atoms.get_potential_energy())

    def gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient at a position.

        Parameters
        ----------
        position
            Flattened coordinates.

        Returns
        -------
        numpy.ndarray
            Flattened gradient, the negative of the forces.
        """
        self.atoms.set_positions(position.reshape(-1, 3))
        return -np.asarray(self.atoms.get_forces(), dtype=float).flatten()

    def function_gradient(self, position: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Evaluate energy and gradient together, which is what the optimisers use.

        Parameters
        ----------
        position
            Flattened coordinates.

        Returns
        -------
        tuple[float, numpy.ndarray]
            Energy in eV and the flattened gradient.
        """
        self.atoms.set_positions(position.reshape(-1, 3))
        forces = np.asarray(self.atoms.get_forces(), dtype=float)
        return float(self.atoms.get_potential_energy()), -forces.flatten()


def _comparer() -> MolecularSimilarity:
    """
    Build the structure comparison used for every distance in this module.

    ``distance_criterion`` is zero deliberately. It is an early-exit threshold:
    the alignment stops optimising once it is under it, so any non-zero value
    returns a distance that is merely below the threshold rather than the
    smallest one. That is harmless for a yes/no test and wrong for a reported
    RMSD, and these distances are reported.

    Returns
    -------
    topsearch.similarity.molecular_similarity.MolecularSimilarity
        Permutation- and inversion-aware comparison.
    """
    return MolecularSimilarity(
        distance_criterion=0.0,
        energy_criterion=1e8,
        weighted=False,
        allow_inversion=True,
    )


def _network(data_dir: Path, molecule: _Molecule) -> tuple:
    """
    Read a molecule's reference network and element order.

    Parameters
    ----------
    data_dir
        Directory holding the extracted archive.
    molecule
        Molecule to read.

    Returns
    -------
    tuple
        The symbols and the ``KineticTransitionNetwork``.
    """
    path = data_dir / molecule.directory
    symbols = read(path / molecule.xyz).get_chemical_symbols()
    network = KineticTransitionNetwork()
    network.read_network(text_path=f"{path}/", text_string="")
    return symbols, network


def _identify(
    position: np.ndarray, minima: list[np.ndarray], symbols: list[str]
) -> int:
    """
    Find which reference minimum a structure corresponds to.

    Needed only for connectivity. Nothing else in reconvergence matches
    anything: a minimum is compared with the reference minimum it started
    from, and a saddle with the saddle it started from. The two structures
    eigenvector following reaches by steepest descent are different -- the
    search produced them, so they carry no index, and identifying them is the
    only way to ask whether the model joined the same pair of basins.

    The nearest reference minimum wins rather than the first one inside the
    threshold. On this reference the closest distinct pair is paracetamol 2-3
    at 0.3404 A, only just clear of ``SAME_STRUCTURE``, so a structure can lie
    within the threshold of both and only the ordering is meaningful.

    Parameters
    ----------
    position
        Flattened coordinates to identify.
    minima
        Reference minima, in network order.
    symbols
        Chemical symbols.

    Returns
    -------
    int
        Index of the matching reference minimum, or -1 if none is within
        ``SAME_STRUCTURE``. A model can descend into a basin the reference
        does not contain, and that is a result rather than an error.
    """
    comparer = _comparer()
    distances = [
        float(
            comparer.closest_distance(
                MolecularCoordinates(symbols, np.asarray(reference).copy()),
                np.asarray(position).copy(),
            )
        )
        for reference in minima
    ]
    best = int(np.argmin(distances))
    return best if distances[best] < SAME_STRUCTURE else -1


def _settle(
    potential: Potential, position: np.ndarray, symbols: list[str]
) -> np.ndarray:
    """
    Finish an endpoint to the same criterion the minima test uses.

    Eigenvector following descends to its own ``steepest_descent_conv_crit``,
    which is looser than ``MINIMISE_CONV_CRIT`` and has to stay that way: the
    same setting governs the subspace minimisation inside every search step, so
    tightening it there would make the whole search far more expensive. The
    endpoints are therefore finished here instead, so that "a minimum" means
    the same thing in this test as in the minima one.

    Measured on malonaldehyde, this moves the endpoints by 80 to 175 further
    iterations and does not change which reference minimum they identify as,
    so it buys consistency rather than a different answer.

    Parameters
    ----------
    potential
        Potential to minimise on.
    position
        Endpoint as eigenvector following returned it.
    symbols
        Chemical symbols.

    Returns
    -------
    numpy.ndarray
        The settled position.
    """
    coords = MolecularCoordinates(symbols, np.asarray(position, dtype=float).copy())
    settled, _, _ = lbfgs.minimise(
        func_grad=potential.function_gradient,
        initial_position=coords.position,
        bounds=coords.bounds,
        conv_crit=MINIMISE_CONV_CRIT,
        n_steps=MINIMISE_STEPS,
    )
    return np.asarray(settled, dtype=float)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_reconverge_minima(mlip: tuple[str, Any]) -> None:
    """
    Relax each reference minimum under the model and record how far it moves.

    A model that agrees with the reference leaves the structure where it is.
    The distance it travels is the metric, and it is not the same question as
    the residual force: a stiff disagreement moves a structure barely at all,
    while a mild one along a soft mode can move it a long way.

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
        set_global_seed(SEED)
        symbols, network = _network(data_dir, molecule)
        potential = _AseCalculatorPotential(symbols, calc)
        comparer = _comparer()

        results: list[Atoms] = []
        for index in tqdm(
            list(network.G.nodes),
            desc=f"{model_name} {molecule.key} minima",
            unit="min",
            disable=not sys.stderr.isatty(),
        ):
            start = network.get_minimum_coords(index)
            coords = MolecularCoordinates(symbols, start.copy())
            position, energy, info = lbfgs.minimise(
                func_grad=potential.function_gradient,
                initial_position=coords.position,
                bounds=coords.bounds,
                conv_crit=MINIMISE_CONV_CRIT,
                n_steps=MINIMISE_STEPS,
            )
            position = np.asarray(position, dtype=float)
            coords.position = start.copy()
            drift = float(comparer.closest_distance(coords, position.copy()))
            coords.position = position.copy()
            hit_budget = info.get("warnflag", 0) == 1

            atoms = Atoms(symbols, position.reshape(-1, 3))
            atoms.info.update(
                molecule=molecule.key,
                kind="minimum",
                label=f"min{index:02d}",
                ref_energy_ev=float(network.get_minimum_energy(index)),
                mlip_energy_ev=float(energy),
                # A minimum that stopped on the iteration limit has not
                # arrived, so how far it travelled is not a measurement of
                # anything. NaN here propagates to the metric and to the
                # model's score, which is the intended behaviour: at a budget
                # of 200 this happened to every structure and the drift still
                # looked entirely reasonable.
                drift=drift if not hit_budget else np.nan,
                valid_minimum=bool(potential.check_valid_minimum(coords)),
                # scipy reports warnflag 1 when it stopped on the iteration
                # limit, which is a protocol result rather than a model one.
                iterations=int(info.get("nit", -1)),
                hit_budget=bool(hit_budget),
            )
            results.append(atoms)

        write(out_dir / f"{molecule.key}_reconverged_minima.extxyz", results)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_reconverge_transition_states(mlip: tuple[str, Any]) -> None:
    """
    Reconverge each reference saddle and record where it lands and what it joins.

    Harder than the minima: a saddle is stationary in only one direction, and
    the model may have no saddle nearby at all. Eigenvector following returns
    the two minima its steepest-descent paths reach, so connectivity is
    recorded here rather than in a separate search -- a model can place a
    saddle correctly and still rewire the network, which changes the kinetics
    while leaving every structure looking right.

    Where the reference records a saddle joining a minimum to itself, a
    degenerate rearrangement between symmetry-equivalent copies, the model is
    counted correct when both of its endpoints match that minimum.

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
        set_global_seed(SEED)
        symbols, network = _network(data_dir, molecule)
        potential = _AseCalculatorPotential(symbols, calc)
        comparer = _comparer()
        minima = [network.get_minimum_coords(i) for i in list(network.G.nodes)]

        results: list[Atoms] = []
        endpoints: list[Atoms] = []
        for plus, minus, edge in tqdm(
            list(network.G.edges),
            desc=f"{model_name} {molecule.key} saddles",
            unit="ts",
            disable=not sys.stderr.isatty(),
        ):
            start = network.get_ts_coords(plus, minus, edge)
            searcher = HybridEigenvectorFollowing(
                potential=potential,
                ts_conv_crit=TS_CONV_CRIT,
                ts_steps=TS_STEPS,
                pushoff=PUSHOFF,
                max_uphill_step_size=MAX_UPHILL_STEP_SIZE,
                positive_eigenvalue_step=POSITIVE_EIGENVALUE_STEP,
                steepest_descent_conv_crit=STEEPEST_DESCENT_CONV_CRIT,
                eigenvalue_conv_crit=EIGENVALUE_CONV_CRIT,
            )
            found, energy, plus_min, _, minus_min, _, _ = searcher.run(
                MolecularCoordinates(symbols, start.copy())
            )

            atoms = Atoms(symbols, np.asarray(start).reshape(-1, 3))
            atoms.info.update(
                molecule=molecule.key,
                kind="transition_state",
                label=f"ts{plus}_{minus}_{edge}",
                connects=f"{plus},{minus}",
                ref_energy_ev=float(network.get_ts_energy(plus, minus, edge)),
                converged=found is not None,
                # Never an empty string: extxyz writes that as a bare
                # "key=" and the reader then takes the NEXT key as its
                # value, silently losing that key.
                failure=searcher.failure or "none",
                steps=int(searcher.n_steps),
                hit_budget=bool(searcher.n_steps >= TS_STEPS),
            )
            if found is not None:
                found = np.asarray(found, dtype=float)
                coords = MolecularCoordinates(symbols, start.copy())
                settled = [
                    _settle(potential, end, symbols) for end in (plus_min, minus_min)
                ]
                reached = sorted(_identify(end, minima, symbols) for end in settled)
                # The endpoints are the evidence for the connectivity result --
                # an endpoint identified as -1 is a basin the reference does not
                # contain, and that claim should be inspectable as a structure.
                for end, index in zip(settled, reached, strict=True):
                    endpoint = Atoms(symbols, end.reshape(-1, 3))
                    endpoint.info.update(
                        molecule=molecule.key,
                        kind="saddle_endpoint",
                        label=f"ts{plus}_{minus}_{edge}",
                        reference_minimum=index,
                    )
                    endpoints.append(endpoint)
                atoms.set_positions(found.reshape(-1, 3))
                atoms.info.update(
                    mlip_energy_ev=float(energy),
                    drift=float(comparer.closest_distance(coords, found.copy())),
                    model_connects=f"{reached[0]},{reached[1]}",
                    connectivity_preserved=reached == sorted((plus, minus)),
                )
            else:
                # The reference geometry is kept so that a failed search still
                # leaves a structure to look at rather than a gap.
                atoms.info.update(
                    mlip_energy_ev=np.nan,
                    drift=np.nan,
                    model_connects="none",
                    connectivity_preserved=False,
                )
            results.append(atoms)

        write(out_dir / f"{molecule.key}_reconverged_saddles.extxyz", results)
        if endpoints:
            write(out_dir / f"{molecule.key}_saddle_endpoints.extxyz", endpoints)


# Displacement for the finite-difference Hessian, in A. Small enough that the
# third-order term stays negligible, large enough that the force difference is
# not dominated by the model's own numerical noise. Each structure costs 6N
# force evaluations, so the whole reference set is a few thousand calls.
HESSIAN_STEP = 1e-3
HESSIAN_FILENAME = "landscape17_hessians.zip"


def _hessian_dir() -> Path:
    """
    Fetch and extract the reference Hessians.

    Returns
    -------
    pathlib.Path
        Directory holding one subdirectory per molecule.
    """
    return download_github_data(filename=HESSIAN_FILENAME, github_uri=DATA_URI) / (
        "landscape17_hessians"
    )


def _reference_hessian(folder: Path, atoms: Atoms) -> np.ndarray:
    """
    Read the reference Hessian for a structure, checking it is that structure.

    The directory is found by name, because the archive's directories carry the
    same labels this module uses. That was not always so: the original archive
    numbered transition states by a position in the edge list that networkx
    does not reproduce, so names could not be trusted and the pairing had to be
    recovered from the coordinates.

    The geometry check is what makes the name safe to rely on. It costs one
    array comparison and turns a renumbered or rebuilt reference from a silent
    comparison against the wrong Hessian into an error.

    Parameters
    ----------
    folder
        Reference directory for this structure.
    atoms
        The structure it should contain.

    Returns
    -------
    numpy.ndarray
        The reference Hessian in eV/A^2.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"No reference Hessian at {folder}")

    reference = read(folder / "geometry.xyz")
    difference = np.abs(reference.get_positions() - atoms.get_positions()).max()
    if difference > 1e-4:
        raise ValueError(
            f"{folder} holds a different structure from the one labelled "
            f"{atoms.info['label']}: coordinates differ by {difference:.3g} A."
        )
    return np.load(folder / "hessian_eV_per_A2.npy")


def _internal_basis(positions: np.ndarray, symbols: list[str]) -> np.ndarray:
    """
    Build a basis spanning internal motion, excluding translation and rotation.

    Six modes sit at essentially zero curvature, and within that near-degenerate
    subspace the individual eigenvectors are arbitrary: comparing them one to
    one produces cosines near zero that look like total disagreement and mean
    nothing. Both Hessians are projected into this basis first, leaving 3N-6
    real modes. The same basis serves both because they share a geometry.

    Parameters
    ----------
    positions
        Atomic positions, N by 3.
    symbols
        Chemical symbols.

    Returns
    -------
    numpy.ndarray
        Columns spanning internal motion.
    """
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    shifted = positions - np.average(positions, axis=0, weights=masses)

    rigid = []
    for axis in range(3):
        translation = np.zeros_like(positions)
        translation[:, axis] = 1.0
        rigid.append(translation.flatten())
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        rigid.append(np.cross(unit, shifted).flatten())

    basis, upper = np.linalg.qr(np.array(rigid).T)
    independent = basis[:, np.abs(np.diag(upper)) > 1e-8]
    left, singular, _ = np.linalg.svd(independent, full_matrices=True)
    return left[:, (singular > 1e-8).sum() :]


def _model_hessian(calc, symbols: list[str], positions: np.ndarray) -> np.ndarray:
    """
    Compute the model's Hessian by central differences of its forces.

    Parameters
    ----------
    calc
        ASE calculator to evaluate with.
    symbols
        Chemical symbols.
    positions
        Atomic positions, N by 3.

    Returns
    -------
    numpy.ndarray
        Symmetrised Hessian in eV/A^2.
    """
    flat = np.asarray(positions, dtype=float).flatten()
    hessian = np.zeros((flat.size, flat.size))
    for i in range(flat.size):
        gradients = []
        for sign in (1, -1):
            shifted = flat.copy()
            shifted[i] += sign * HESSIAN_STEP
            atoms = Atoms(symbols, shifted.reshape(-1, 3))
            atoms.info.update(charge=CHARGE, spin=SPIN)
            atoms.calc = calc
            gradients.append(-np.asarray(atoms.get_forces(), dtype=float).flatten())
        hessian[i] = (gradients[0] - gradients[1]) / (2 * HESSIAN_STEP)
    # The two halves differ only by finite-difference error; a Hessian is
    # symmetric by construction, so averaging them is free accuracy.
    return 0.5 * (hessian + hessian.T)


def _match_modes(
    reference: np.ndarray, model: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pair every reference mode with a model mode, one to one, by overlap.

    An eigenvalue window cannot do this job. A model whose stiff modes come
    back uniformly softer than DFT -- GFN2-xTB is 15 to 20% soft on this
    reference -- puts the correct partner outside any window tight enough to
    be meaningful, and the mode is scored zero although the model has its
    direction right. Matching on the overlap itself is indifferent to a
    systematic scaling, which is then reported separately instead of silently
    destroying the comparison.

    One to one also handles degeneracy: modes free to mix within a degenerate
    set are assigned among themselves rather than all claiming one partner.

    Parameters
    ----------
    reference
        Reference eigenvectors as columns.
    model
        Model eigenvectors as columns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The matched overlap for each reference mode, and the model mode index
        it was matched to.
    """
    overlap = np.abs(reference.T @ model)
    rows, columns = linear_sum_assignment(-overlap)
    matched = np.zeros(reference.shape[1])
    partner = np.zeros(reference.shape[1], dtype=int)
    matched[rows] = overlap[rows, columns]
    partner[rows] = columns
    return matched, partner


@pytest.mark.parametrize("mlip", MODELS.items())
def test_hessians(mlip: tuple[str, Any]) -> None:
    """
    Compare the model's curvature with DFT at each reference stationary point.

    Both Hessians are evaluated at the same DFT geometry, so this measures the
    curvature the model assigns to the reference structure rather than
    curvature at some nearby structure the model happened to find. That keeps
    it a test of the potential and not of the search.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    hessian_dir = _hessian_dir()
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = _data_dir()

    for molecule in MOLECULES:
        results: list[Atoms] = []
        frames = _stationary_points(data_dir, molecule)
        for atoms in tqdm(
            frames,
            desc=f"{model_name} {molecule.key} hessians",
            unit="pt",
            disable=not sys.stderr.isatty(),
        ):
            folder = hessian_dir / molecule.key / atoms.info["label"]
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            reference = _reference_hessian(folder, atoms)

            out_atoms = Atoms(symbols, positions)
            out_atoms.info.update(
                molecule=molecule.key,
                kind=atoms.info["kind"],
                label=atoms.info["label"],
            )
            try:
                predicted = _model_hessian(calc, symbols, positions)
            except Exception as exc:
                warn(f"Hessian failed on {atoms.info['label']}: {exc}", stacklevel=2)
                out_atoms.info.update(
                    mode_overlap_median=np.nan,
                    reaction_coordinate_overlap=np.nan,
                )
                results.append(out_atoms)
                continue

            basis = _internal_basis(positions, symbols)
            ref_values, ref_vectors = np.linalg.eigh(basis.T @ reference @ basis)
            model_values, model_vectors = np.linalg.eigh(basis.T @ predicted @ basis)
            matched, partner = _match_modes(ref_vectors, model_vectors)

            paired = model_values[partner]

            out_atoms.info.update(
                mode_overlap_median=float(np.median(matched)),
                mode_overlap_min=float(matched.min()),
                # The eigenvalues themselves, paired mode by mode, rather than
                # a scaling derived from them. How to summarise the curvature
                # is a real choice -- a least-squares slope is dominated by the
                # stiffest modes, which on this reference span four orders of
                # magnitude -- and keeping the pairs means that choice can be
                # revisited in the analysis without recomputing any Hessian.
                reference_eigenvalues=ref_values,
                model_eigenvalues=paired,
                # Diagnostic, not a metric. Counting negative eigenvalues is
                # unreliable near zero: paracetamol's softest torsions sit at
                # a few times 1e-3 eV/A^2 and three reference structures have
                # one dip just below zero -- min01 at -7e-4, and two saddles
                # with a spurious -1e-3 alongside their genuine -0.5. The real
                # imaginary mode is still the most negative in every case, so
                # the reaction coordinate below is unaffected.
                n_negative_reference=int((ref_values < 0).sum()),
                n_negative_model=int((model_values < 0).sum()),
                # At a transition state the lowest mode is the reaction
                # coordinate; whether the model agrees about its direction is
                # a sharper question than the average mode. Stored raw: whether
                # to count it when the model has no imaginary mode there is a
                # judgement, and the eigenvalues beside it let the analysis
                # make that judgement without recomputing a Hessian.
                reaction_coordinate_overlap=(
                    float(matched[0])
                    if atoms.info["kind"] == "transition_state"
                    else np.nan
                ),
            )
            results.append(out_atoms)

        if results:
            write(out_dir / f"{molecule.key}_hessians.extxyz", results)
