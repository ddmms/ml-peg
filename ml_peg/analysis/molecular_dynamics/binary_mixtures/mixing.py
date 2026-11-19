"""
Shared calculations for the binary mixtures benchmark.

Used by ``analyse_binary_mixtures.py`` (MLIPs) and by
``liquid_mixtures_inputs.summarise_sage`` in the separate input-generation
repository, https://github.com/fjclark/ml-peg-liquid-mixtures-input-generation
(the Sage reference runs), so both use the same estimators. That
repository imports this module, so treat its public names as an API. Nothing
here loads models or reads benchmark constants from the calc module.

Box means and their standard errors come from
:func:`ml_peg.analysis.utils.timeseries.estimate_mean`, applied to every logged
sample. Block averages are used only for the drift flag, never for
uncertainties.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms, units
from ase.neighborlist import natural_cutoffs, neighbor_list
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from scipy import constants, stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ml_peg.analysis.utils.timeseries import TimeSeriesEstimate, estimate_mean

PRESSURE_EV_PER_A3 = constants.atm * units.Pascal  # the barostat pressure
# Factors that convert from ASE units (eV, Å) by multiplication
EV_TO_KJ_PER_MOL = units.mol / units.kJ
A3_TO_CM3_PER_MOL = units.mol / (constants.centi * units.m) ** 3
N_BLOCKS = 10
MIN_N_EFF = 100
FRAME_EVERY_PS = 100.0
BOND_CUTOFF_MULT = 1.2
NEIGHBOUR_SHELL = 1.5  # like-neighbour cutoff, in mean intermolecular spacings
OBSERVABLES = ("h", "density", "v_molar")


@dataclass
class BoxResult:
    """
    Statistics of one production run.

    Attributes
    ----------
    estimates
        Mean and uncertainty of each observable in ``OBSERVABLES``.
    p_drift
        Unadjusted p-value of the drift test per observable.
    run_ps
        Time of the last logged sample.
    """

    estimates: dict[str, TimeSeriesEstimate]
    p_drift: dict[str, float] = field(default_factory=dict)
    run_ps: float = np.nan


def read_production(
    csv_path: Path, equilibration_ps: float, total_ps: float
) -> pd.DataFrame:
    """
    Read the production part of a finished run's scalar log.

    Parameters
    ----------
    csv_path
        Scalar CSV written by the calc (or the Sage runner).
    equilibration_ps
        Equilibration time to discard.
    total_ps
        Expected total run length.

    Returns
    -------
    pd.DataFrame
        Rows with ``time_ps > equilibration_ps``.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    ValueError
        If the run did not reach ``total_ps``.
    """
    try:
        data = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        data = pd.DataFrame()
    reached_ps = data.time_ps.iloc[-1] if not data.empty else 0.0
    if reached_ps < total_ps:
        raise ValueError(
            f"{csv_path} is incomplete: reached {reached_ps:g} of {total_ps:g} ps"
        )
    return data[data.time_ps > equilibration_ps].reset_index(drop=True)


def observables(prod: pd.DataFrame, n_molecules: int) -> dict[str, np.ndarray]:
    """
    Per-sample observables of a production run.

    Parameters
    ----------
    prod
        Production rows of the scalar log.
    n_molecules
        Number of molecules in the box.

    Returns
    -------
    dict[str, np.ndarray]
        ``h`` = (Epot + pV) per molecule in kJ/mol (kinetic energy cancels in
        ΔHmix), ``density`` in g/cm³ and ``v_molar`` in cm³/mol.
    """
    volume = prod.volume_A3.to_numpy()
    enthalpy = prod.epot_eV.to_numpy() + (PRESSURE_EV_PER_A3 * volume)
    return {
        "h": (enthalpy / n_molecules) * EV_TO_KJ_PER_MOL,
        "density": prod.density_g_cm3.to_numpy(),
        "v_molar": (volume / n_molecules) * A3_TO_CM3_PER_MOL,
    }


def drift_test(t: np.ndarray, y: np.ndarray) -> float:
    """
    Test block means for drift.

    Parameters
    ----------
    t
        Times in ps.
    y
        Values.

    Returns
    -------
    float
        P-value of a nonzero slope of the block means.
    """
    # Split the run into N_BLOCKS consecutive blocks and average each one
    block_t = np.array([b.mean() for b in np.array_split(t, N_BLOCKS)])
    block_y = np.array([b.mean() for b in np.array_split(y, N_BLOCKS)])
    # Drift: is the slope of a straight line through the block means nonzero?
    return float(stats.linregress(block_t, block_y).pvalue)


def box_result(
    prod: pd.DataFrame,
    n_molecules: int,
    label: str | None = None,
) -> BoxResult:
    """
    Means, uncertainties and convergence tests of one production run.

    Parameters
    ----------
    prod
        Production rows of the scalar log (from :func:`read_production`).
    n_molecules
        Number of molecules in the box.
    label
        Label for warnings.

    Returns
    -------
    BoxResult
        Statistics of the run.
    """
    t = prod.time_ps.to_numpy()
    result = BoxResult(estimates={}, run_ps=float(t[-1]))
    for name, y in observables(prod, n_molecules).items():
        estimate = estimate_mean(
            y,
            min_n_eff=MIN_N_EFF,
            label=f"{label} {name}" if label else name,
        )
        result.estimates[name] = estimate
        result.p_drift[name] = drift_test(t, y)
    return result


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """
    Holm-Bonferroni adjusted p-values.

    Parameters
    ----------
    pvalues
        Raw p-values.

    Returns
    -------
    np.ndarray
        Adjusted p-values in the original order. NaN p-values (failed tests)
        stay NaN and do not count towards the number of tests.
    """
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def mixture_box_id(system_id: str, x_box: float) -> str:
    """
    Name of the mixture box for a system and composition.

    Parameters
    ----------
    system_id
        System identifier.
    x_box
        Mole fraction of the first component, as in ``systems.csv``.

    Returns
    -------
    str
        Box identifier, e.g. ``methanol_water_x500``.
    """
    return f"{system_id}_x{round(x_box * 1000):03d}"


def mixing_matrix(
    systems: pd.DataFrame, boxes: pd.DataFrame, box_ids: Sequence[str]
) -> np.ndarray:
    """
    Linear map from box means to mixing properties.

    Row i gives property_mix - x·property_a - (1 - x)·property_b for row i of
    ``systems``, with x the exact composition of the mixture box. Applied to
    per-molecule enthalpies this is ΔHmix; applied to molar volumes it is V^E.

    Parameters
    ----------
    systems
        ``systems.csv`` (columns ``system_id``, ``a``, ``b``, ``x_box``).
    boxes
        ``boxes.csv`` (columns ``box_id``, ``kind``, ``smiles``,
        ``temperature``, ``x``).
    box_ids
        Column order of the matrix.

    Returns
    -------
    np.ndarray
        Matrix with shape (len(systems), len(box_ids)).
    """
    column = {box_id: i for i, box_id in enumerate(box_ids)}
    by_id = boxes.set_index("box_id")
    pure = boxes[boxes.kind == "pure"].set_index(["smiles", "temperature"]).box_id
    matrix = np.zeros((len(systems), len(box_ids)))
    for i, system in enumerate(systems.itertuples()):
        mix_id = mixture_box_id(system.system_id, system.x_box)
        mix = by_id.loc[mix_id]
        x = float(mix.x)
        matrix[i, column[mix_id]] += 1.0
        matrix[i, column[pure.loc[(system.a, mix.temperature)]]] -= x
        matrix[i, column[pure.loc[(system.b, mix.temperature)]]] -= 1.0 - x
    return matrix


def combine(
    weights: Sequence[float], estimates: Sequence[TimeSeriesEstimate]
) -> TimeSeriesEstimate:
    """
    Estimate of a linear combination of independent box means.

    Parameters
    ----------
    weights
        Coefficient of each estimate (zeros are skipped).
    estimates
        Box estimates, in the same order.

    Returns
    -------
    TimeSeriesEstimate
        Combined mean, standard error (added in quadrature), the smallest
        effective sample size of the terms used, and ci_95 = 1.96 sem.
    """
    terms = [(w, e) for w, e in zip(weights, estimates, strict=True) if w != 0]
    mean = float(sum(w * e.mean for w, e in terms))
    sem = float(np.sqrt(sum((w * e.sem) ** 2 for w, e in terms)))
    return TimeSeriesEstimate(
        mean=mean,
        sem=sem,
        n_eff=float(min(e.n_eff for _, e in terms)),
        ci_95=1.96 * sem,
    )


def find_molecules(atoms: Atoms) -> np.ndarray:
    """
    Label the covalently bonded molecules of a periodic box.

    Parameters
    ----------
    atoms
        Periodic box.

    Returns
    -------
    np.ndarray
        Molecule label of every atom. Labels are numbered in order of each
        molecule's lowest atom index.
    """
    cutoffs = natural_cutoffs(atoms, mult=BOND_CUTOFF_MULT)
    i, j = neighbor_list("ij", atoms, cutoffs)
    n_atoms = len(atoms)
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n_atoms, n_atoms))
    _, labels = connected_components(graph, directed=False)
    # Relabel so molecule k is the k-th to appear in atom order
    _, first = np.unique(labels, return_index=True)  # lowest atom of each molecule
    rank = np.empty_like(first)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[labels]


def same_molecules(reference: np.ndarray, labels: np.ndarray) -> bool:
    """
    Check whether two molecule labellings describe the same molecules.

    Parameters
    ----------
    reference
        Labels of the reference frame (from :func:`find_molecules`).
    labels
        Labels of another frame.

    Returns
    -------
    bool
        ``True`` if every molecule has exactly the same atoms.
    """
    return bool(np.array_equal(reference, labels))


def _minimum_image(vectors: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Shift vectors to their nearest periodic copy.

    Rounding fractional coordinates gives the minimum image exactly for
    orthorhombic cells, such as the cubic benchmark boxes.

    Parameters
    ----------
    vectors
        Vectors with last dimension 3.
    cell
        Cell matrix, with the lattice vectors as rows.

    Returns
    -------
    np.ndarray
        Vectors of the same shape, each within half a box length per axis.
    """
    fractional = vectors @ np.linalg.inv(cell)
    return (fractional - np.round(fractional)) @ cell


def molecule_centres(atoms: Atoms, labels: np.ndarray) -> np.ndarray:
    """
    Calculate molecular centres of mass, unwrapped across periodic boundaries.

    Each molecule is made whole by taking every atom's minimum image relative
    to the molecule's first atom.

    Parameters
    ----------
    atoms
        Periodic box.
    labels
        Molecule label of every atom.

    Returns
    -------
    np.ndarray
        Centres with shape (n_molecules, 3).
    """
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    _, first = np.unique(labels, return_index=True)  # first atom of each molecule
    # Each atom's position relative to its molecule's first atom, taking the
    # nearest periodic copy so that molecules split by the boundary are whole
    offsets = _minimum_image(positions - positions[first[labels]], atoms.cell.array)
    # Mass-weighted mean offset of each molecule's atoms
    weighted_sum = np.zeros((len(first), 3))
    np.add.at(weighted_sum, labels, masses[:, None] * offsets)
    molecule_mass = np.bincount(labels, weights=masses)
    return positions[first] + weighted_sum / molecule_mass[:, None]


def like_fraction(atoms: Atoms, labels: np.ndarray, is_a: np.ndarray) -> float:
    """
    Mean fraction of a molecule's neighbours that are the same component.

    Neighbours are molecules whose centres of mass are within
    ``NEIGHBOUR_SHELL`` mean intermolecular spacings, (V/N)^(1/3), so the
    cutoff scales with molecular size. Demixing raises the value above that of
    a well-mixed reference.

    Parameters
    ----------
    atoms
        Periodic box.
    labels
        Molecule label of every atom.
    is_a
        Whether each molecule is component a, with shape (n_molecules,).

    Returns
    -------
    float
        Like-neighbour fraction averaged over molecules with neighbours.
    """
    centres = molecule_centres(atoms, labels)
    n_molecules = len(centres)
    spacing = (atoms.get_volume() / n_molecules) ** (1 / 3)
    # Distances between every pair of centres, to the nearest periodic copy
    separations = centres[:, None, :] - centres[None, :, :]
    distances = np.linalg.norm(_minimum_image(separations, atoms.cell.array), axis=-1)
    # neighbours[i, j]: molecule j is within the cutoff of molecule i (i != j)
    neighbours = distances < NEIGHBOUR_SHELL * spacing
    np.fill_diagonal(neighbours, False)
    same_component = is_a[:, None] == is_a[None, :]
    n_neighbours = neighbours.sum(axis=1)
    n_like = (neighbours & same_component).sum(axis=1)
    has_neighbours = n_neighbours > 0
    return float(np.mean(n_like[has_neighbours] / n_neighbours[has_neighbours]))


def _smiles_formula(smiles: str) -> str:
    """
    Get the Hill formula of a molecule, as ASE writes it.

    Parameters
    ----------
    smiles
        SMILES of the molecule.

    Returns
    -------
    str
        Hill formula, e.g. ``C2H6O``.
    """
    return CalcMolFormula(Chem.MolFromSmiles(smiles))


def component_a_molecules(atoms: Atoms, labels: np.ndarray) -> np.ndarray:
    """
    Which molecules belong to component a, identified by chemical formula.

    Each molecule's formula is compared with those of ``smiles_a`` and
    ``smiles_b`` in ``atoms.info``, so the order of the molecules in the box
    does not matter.

    Parameters
    ----------
    atoms
        Mixture box, with ``smiles_a``, ``smiles_b``, ``n_a`` and ``n_b`` in
        ``atoms.info``.
    labels
        Molecule label of every atom.

    Returns
    -------
    np.ndarray
        Boolean array with shape (n_molecules,).

    Raises
    ------
    ValueError
        If the components share a formula, a molecule matches neither, or the
        molecule counts differ from ``n_a`` and ``n_b``.
    """
    formula_a = _smiles_formula(atoms.info["smiles_a"])
    formula_b = _smiles_formula(atoms.info["smiles_b"])
    if formula_a == formula_b:
        raise ValueError(f"Both components are {formula_a}, so cannot be told apart")
    formulas = np.array(
        [
            atoms[labels == label].get_chemical_formula(mode="hill")
            for label in range(labels.max() + 1)
        ]
    )
    unknown = set(formulas) - {formula_a, formula_b}
    if unknown:
        raise ValueError(f"Molecules match neither component: {sorted(unknown)}")
    is_a = formulas == formula_a
    counts = (int(is_a.sum()), int((~is_a).sum()))
    expected = (int(atoms.info["n_a"]), int(atoms.info["n_b"]))
    if counts != expected:
        raise ValueError(f"Found {counts} molecules of (a, b), expected {expected}")
    return is_a


def frame_checks(
    frames: Sequence[Atoms], reference: Atoms, mixture: bool
) -> tuple[bool, float]:
    """
    Reaction and demixing checks over a set of frames.

    Parameters
    ----------
    frames
        Frames to check (e.g. every 100 ps of production).
    reference
        Starting box whose molecules define the expected bonding, with the
        box metadata in ``info``.
    mixture
        Whether to compute the like-neighbour fraction.

    Returns
    -------
    tuple[bool, float]
        Whether any frame's molecules differ from the reference, and the mean
        like-neighbour fraction (NaN for pure boxes or no frames).
    """
    ref_labels = find_molecules(reference)
    is_a = component_a_molecules(reference, ref_labels) if mixture else None
    reacted = False
    fractions = []
    for frame in frames:
        reacted |= not same_molecules(ref_labels, find_molecules(frame))
        if mixture:
            fractions.append(like_fraction(frame, ref_labels, is_a))
    return reacted, float(np.mean(fractions)) if fractions else np.nan
