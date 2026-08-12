"""Analyse HPHT_CH4_H2O benchmark."""

from __future__ import annotations

from pathlib import Path
import warnings

from ase.io import iread
from ase.neighborlist import (
    NeighborList,
    NewPrimitiveNeighborList,
    neighbor_list,
)
import numpy as np
import plotly.graph_objects as go
import pytest
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "molecular_reactions" / "HPHT_CH4_H2O" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "HPHT_CH4_H2O"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)
OUT_PATH.mkdir(parents=True, exist_ok=True)


def make_fes(data, start, end, n_bins, t):
    """
    Calculate a free energy profile from a histogram of raw reaction coordinate values.

    Create a numpy histogram from raw reaction coordinate values.
    Then computes free energy profile where the minimum is set to
    zero together with centered bin values.
    Free energy value is set to NaN if the probability is 0 to
    avoid infinite free energy values.

    Parameters
    ----------
    data : list or array
        Raw reaction coordinate values (in Å).
    start : float
        First bin boundary.
    end : float
        Last bin boundary.
    n_bins : int
        Total number of desired bins.
    t : float
        Temperature of the simulation (in K).

    Returns
    -------
    bin_centers : ndarray
        Centered bins values (in Å).
    f : ndarray
        Free energy values (in kJ/mol).
    """
    hist, bin_edges = np.histogram(data, bins=n_bins, range=(start, end))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = np.sum(hist)
    if total == 0:
        return bin_centers, np.full(n_bins, np.nan)
    prob = hist / total
    k = 8.314
    mask = prob != 0
    f = np.full_like(prob, np.nan)
    f[mask] = -k * t * np.log(prob[mask])
    f -= np.nanmin(f)
    return bin_centers, f


def build_h_heavy_neighborlist_simple(atoms, cutoff_h_to_heavy):
    """
    Build a neighborlist giving H-heavy atom pairs within a cutoff distance criteria.

    This stands as a search for H atoms in a sphere of radius cutoff_h_to_heavy
    and centered on heavy atoms (C,O)
    It ensures :
      - H-heavy connections via bothways
      - H-H connections are impossible
      - heavy-heavy connections are possible but will be ignored
        in the pratical use of the function.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure used to build the neighborlist.
    cutoff_h_to_heavy : float
        Distance cutoff (in Å) to use.

    Returns
    -------
    ase.neighborlist.NeighborList
        Neighborlist object containing the defined atomic connections.
    """
    symbols = atoms.get_chemical_symbols()
    cutoffs = []

    for sym in symbols:
        if sym == "H":
            cutoffs.append(0.0)
        elif sym in ("C", "O"):
            cutoffs.append(cutoff_h_to_heavy)
        else:
            cutoffs.append(0.0)

    nl = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=True)
    nl.update(atoms)
    return nl


def second_nearest_heavy_outside_molecule(h, atoms, nl, labels):
    """
    Find the closest heavy atom to a given hydrogen atom, outside of its own molecule.

    The identified atom corresponds to the second nearest
    heavy-atom neighbor of the hydrogen atom.

    Parameters
    ----------
    h : int
        Hydrogen atom index to search for.
    atoms : ase.Atoms
        Atom structure used by the function.
    nl : ase.neighborlist.NeighborList
        Neighborlist object.
    labels : list
        List of atom indices belonging to each molecule.

    Returns
    -------
    heavy2 : int
        Index of the second nearest heavy-atom neighbor of hydrogen atom h.
    min_dist : float
        Distance between h and heavy2 (in Å).
    """
    mol_h = labels[h]

    neighbors, offsets = nl.get_neighbors(h)

    min_dist = float("inf")
    heavy2 = None
    for idx, shift in zip(neighbors, offsets, strict=True):
        if labels[idx] == mol_h:
            continue

        dr = atoms.positions[h] - (atoms.positions[idx] + shift @ atoms.cell)
        dist = np.linalg.norm(dr)

        if dist < min_dist:
            min_dist = dist
            heavy2 = idx

    return heavy2, min_dist


def build_connectivity_matrix_coh_hh_force(
    atoms, dcc=1.8, doo=1.6, dco=1.7, dch=1.4, doh=1.3, dhh=1.2
):
    """
    Build a C/O/H connectivity matrix matching the molecular recognition's requirements.

    More precisely, the procedure ensures that:
        1. H atoms are connected to their nearest heavy atom
           with respect to pair cutoff values (dCH,dOH).
        2. Heavy atoms are connected to each other
           with respect to associated pair cutoff values (dCC,dCO,dOO).
        3. H atoms left without nearest heavy atoms after the step 1 (H_orphan list)
           are connected to each other, standing as a search of H2 molecules.
        4. The remaining H atoms still not connected to anyone (H_lonely list)
           are attached to the nearest heavy atoms without using dCH/dOH cutoffs.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure used to build connectivity matrix.
    dcc : float
        Distance based C-C pair cutoffs.
    doo : float
        Distance based O-O pair cutoffs.
    dco : float
        Distance based C-O pair cutoffs.
    dch : float
        Distance based C-H pair cutoffs.
    doh : float
        Distance based O-H pair cutoffs.
    dhh : float
        Distance based H-H pair cutoffs.

    Returns
    -------
    matrix : spicy.sparse.lil_matrix
        Sparse C/O/H connectivity matrix.
    h_to_heavy : dict
        Mapping between hydrogen atom indices and their assigned heavy atom index.
    h_orphan : list
        Hydrogen atoms indices not connected to any heavy atom within dCH/dOH cutoffs.
    h_lonely : list
        Hydrogen atoms indices from H_orphan that are forcibly connected
        to the nearest heavy atom.
    """
    n_atoms = len(atoms)
    idx_h = [i for i, a in enumerate(atoms) if a.symbol == "H"]
    idx_heavy = [i for i, a in enumerate(atoms) if a.symbol in ("C", "O")]
    pair_cutoffs = {
        ("C", "C"): dcc,
        ("O", "O"): doo,
        ("C", "O"): dco,
        ("O", "C"): dco,
        ("C", "H"): dch,
        ("H", "C"): dch,
        ("O", "H"): doh,
        ("H", "O"): doh,
        ("H", "H"): dhh,
    }
    matrix = lil_matrix((n_atoms, n_atoms), dtype=int)
    # ------------------------------------------------------------------------------
    # Search of H atoms contained in dch/doh radius spheres centered on heavy atoms.
    # ------------------------------------------------------------------------------
    symbols = atoms.get_chemical_symbols()
    cutoffs = []

    for sym in symbols:
        if sym == "H":
            cutoffs.append(0.0)
        elif sym == "C":
            cutoffs.append(pair_cutoffs.get(("C", "H"), 0.0))
        elif sym == "O":
            cutoffs.append(pair_cutoffs.get(("O", "H"), 0.0))

    nl_h = NeighborList(
        cutoffs,
        skin=0,
        self_interaction=False,
        bothways=True,
        primitive=NewPrimitiveNeighborList,
    )
    nl_h.update(atoms)
    # -----------------------------------------------------------------------
    # Connection of each H atoms contained in dch/doh radius spheres
    # with its nearest neighbor. Other H atoms are sent to the h_oprhan list.
    # -----------------------------------------------------------------------
    h_to_heavy = {}
    h_orphan = []
    for h in idx_h:
        neighbors, offsets = nl_h.get_neighbors(h)
        min_dist = float("inf")
        nearest = None
        if len(neighbors) == 0:
            h_orphan.append(h)
        else:
            for idx, shift in zip(neighbors, offsets, strict=True):
                dr = atoms.positions[h] - (atoms.positions[idx] + shift @ atoms.cell)
                dist = np.linalg.norm(dr)

                if dist < min_dist:
                    min_dist = dist
                    nearest = idx
            h_to_heavy[h] = nearest

    for h, heavy in h_to_heavy.items():
        matrix[h, heavy] = 1
        matrix[heavy, h] = 1
    # ------------------------------------------------------------------------
    # Manual connection between heavy atoms with repsect to heavy pair cutoffs
    # from a global neighbor list.
    # ------------------------------------------------------------------------
    max_cutoff = max(pair_cutoffs.values())
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff=max_cutoff)
    symbols = atoms.get_chemical_symbols()

    for idx in range(len(i_list)):
        i = i_list[idx]
        j = j_list[idx]

        sym_i = symbols[i]
        sym_j = symbols[j]

        if sym_i not in ("C", "O") or sym_j not in ("C", "O"):
            continue

        pair = tuple(sorted((sym_i, sym_j)))
        allowed = pair_cutoffs.get(pair, 0.0)

        if d_list[idx] <= allowed:
            matrix[i, j] = 1
            matrix[j, i] = 1
    # -------------------------------------------------------------------
    # Connection between orphan H atoms with respect to dhh cutoff.
    # Remaining unconnected orphan H atoms are sent to the h_lonely list.
    # -------------------------------------------------------------------
    h_lonely = []
    if h_orphan:
        cutoffs_hh = np.zeros(n_atoms)
        for h in h_orphan:
            cutoffs_hh[h] = dhh
        nl_hh = NeighborList(cutoffs_hh, skin=0, self_interaction=False, bothways=True)
        nl_hh.update(atoms)
        for h in h_orphan:
            neighbors, offsets = nl_hh.get_neighbors(h)
            has_h_neighbor = False
            for i, n in enumerate(neighbors):
                if n in h_orphan:
                    shift = offsets[i] @ atoms.cell
                    dr = atoms.positions[h] - (atoms.positions[n] + shift)
                    if np.linalg.norm(dr) <= dhh:
                        matrix[h, n] = 1
                        matrix[n, h] = 1
                        has_h_neighbor = True
            if not has_h_neighbor:
                h_lonely.append(h)
    # -----------------------------------------------------------------------------
    # Manual connection of lonely H atoms to the nearest heavy atom without cutoff.
    # Return of matrix, h_to_heavy dictionnary, and h_orphan and h_lonely lists.
    # -----------------------------------------------------------------------------
    for h in h_lonely:
        min_dist = float("inf")
        nearest_heavy = None
        for n in idx_heavy:
            dist = atoms.get_distance(h, n, mic=True)
            if dist < min_dist:
                min_dist = dist
                nearest_heavy = n
        if nearest_heavy is not None:
            matrix[h, nearest_heavy] = 1
            matrix[nearest_heavy, h] = 1

            h_to_heavy[h] = nearest_heavy

    return matrix, h_to_heavy, h_orphan, h_lonely


def compute_fes(extxyz_file, t):
    """
    Compute the free energy profile of proton hopping between H3O+ and CH4 species.

    The function performs a molecular recognition procedure
    on the input trajectory, computes the reaction coordinate for each frame,
    and converts its distribution into a free energy profile.

    Parameters
    ----------
    extxyz_file : str
        Input trajectory file obtained from the calculation part.
    t : int
        Temperature of the simulation (in K).

    Returns
    -------
    bins : ndarray
        Reaction coordinate values.
    f : ndarray
        Free energy values corresponding to 'bins'.
    """
    cutoff_h_to_heavy = 3.0
    frame_max = 100000
    coord = []
    for frame_idx, atoms in enumerate(iread(extxyz_file, format="extxyz")):
        if frame_idx == frame_max:
            break
        matrix, h_to_heavy, h_orphan, h_lonely = build_connectivity_matrix_coh_hh_force(
            atoms
        )
        n_mol, labels = connected_components(matrix)
        molecules = []
        for mol_idx in range(n_mol):
            atom_indices = tuple(int(i) for i in np.where(labels == mol_idx)[0])
            counts = {"C": 0, "O": 0, "H": 0}
            for i in atom_indices:
                sym = atoms[i].symbol
                if sym in counts:
                    counts[sym] += 1
            molecules.append(
                {
                    "atoms": atom_indices,
                    "nC": counts["C"],
                    "nH": counts["H"],
                    "nO": counts["O"],
                    "label": f"C{counts['C']}H{counts['H']}O{counts['O']}",
                }
            )

        nl = build_h_heavy_neighborlist_simple(atoms, cutoff_h_to_heavy)

        for h in h_to_heavy:
            h_mol_idx = labels[h]
            if molecules[h_mol_idx]["label"] == "C0H3O1":
                heavy2, dist_heavy2 = second_nearest_heavy_outside_molecule(
                    h, atoms, nl, labels
                )
                if molecules[labels[heavy2]]["label"] == "C1H4O0":
                    dist_heavy = atoms.get_distance(h, h_to_heavy[h], mic=True)
                    coord.append(dist_heavy - dist_heavy2)
            if molecules[h_mol_idx]["label"] == "C1H5O0":
                heavy2, dist_heavy2 = second_nearest_heavy_outside_molecule(
                    h, atoms, nl, labels
                )
                if molecules[labels[heavy2]]["label"] == "C0H2O1":
                    dist_heavy = atoms.get_distance(h, h_to_heavy[h], mic=True)
                    coord.append(dist_heavy2 - dist_heavy)

    bins, f = make_fes(coord, start=-1.5, end=1.5, n_bins=200, t=t)
    return bins, f


def load_reference_fes(structure_name):
    """
    Load the reference free energy profile of the input structure.

    Parameters
    ----------
    structure_name : str
        Name of the referene structure.

    Returns
    -------
    bins : ndarray
        Reaction coordinate values.
    f : ndarray
        Free energy values corresponding to 'bins'.
    """
    ref_dir = (
        download_s3_data(
            key="inputs/molecular_reactions/HPHT_CH4_H2O/HPHT_CH4_H2O.zip",
            filename="HPHT_CH4_H2O.zip",
        )
        / "HPHT_CH4_H2O"
    )
    ref_file = ref_dir / f"{structure_name}.data"

    if not ref_file.exists():
        raise FileNotFoundError(f"Missing reference file {ref_file}")

    data = np.loadtxt(ref_file)

    bins = data[:, 0]
    f = data[:, 1]

    return bins, f


@pytest.fixture
def free_energy_profiles():
    """
    Generate the free energy profiles of all CH4/H2O systems for all MODELS.

    Write free energy profile in .data files
    and generate associated plots in .json files in the application path
    to be used by the application part.

    Returns
    -------
    dict
        Dictionary where:
            - ``"x"`` contains the reaction coordinate values (`ndarray`);
            - ``"ref"`` contains the reference free energy profiles (`list[ndarray]`);
            - each model name maps to a list of predicted free energy profiles
            (`list[ndarray]`).
    """
    results = {
        "x": None,
        "ref": [],
    } | {model: [] for model in MODELS}
    structures = get_structures_names()
    for structure_name in structures:
        ref_bins, ref_f = load_reference_fes(structure_name)
        save_ref_path = OUT_PATH / f"{structure_name}.data"
        np.savetxt(save_ref_path, np.column_stack((ref_bins, ref_f)))

        results["ref"].append(ref_f)
        if results["x"] is None:
            results["x"] = ref_bins

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            warnings.warn(
                f"Missing model directory {model_name}."
                "Filling all free energy profiles with NaN.",
                stacklevel=2,
            )
            for _ in structures:
                results[model_name].append(
                    np.full_like(results["x"], np.nan, dtype=float)
                )
            continue
        for structure_name in structures:
            xyz_file = model_dir / f"{structure_name}.extxyz"
            if xyz_file.exists():
                bins, f_model = compute_fes(xyz_file, 3000)
            else:
                warnings.warn(
                    f"Missing trajectory file {xyz_file}."
                    "Filling free energy profile with NaN.",
                    stacklevel=2,
                )
                bins = results["x"]
                f_model = np.full_like(bins, np.nan, dtype=float)

            save_path = OUT_PATH / model_name / f"{structure_name}.data"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(save_path, np.column_stack((bins, f_model)))
            results[model_name].append(f_model)
            if model_name == MODELS[-1]:
                fig = go.Figure()

                ref_file = OUT_PATH / f"{structure_name}.data"
                ref_data = np.loadtxt(ref_file)

                fig.add_trace(
                    go.Scatter(
                        x=ref_data[:, 0],
                        y=ref_data[:, 1],
                        mode="lines",
                        name="Reference",
                    )
                )

                for model in MODELS:
                    model_file = OUT_PATH / model / f"{structure_name}.data"
                    if not model_file.exists():
                        continue

                    model_data = np.loadtxt(model_file)

                    fig.add_trace(
                        go.Scatter(
                            x=model_data[:, 0],
                            y=model_data[:, 1],
                            mode="lines",
                            name=model,
                        )
                    )
                fig.update_layout(
                    title=f"Free Energy Profile - {structure_name}",
                    xaxis_title="Reaction coordinate",
                    yaxis_title="Free energy (kJ/mol)",
                    template="plotly_white",
                )

                plot_path = OUT_PATH / "fes_plots"
                plot_path.mkdir(parents=True, exist_ok=True)
                fig.write_json(plot_path / f"{structure_name}.json")

    return results


def reaction_free_energy(f, bins):
    """
    Compute free energy of reaction and free energy barrier while ignoring NaN values.

    The reaction free energy is defined as F(product) - F(reactant),
    and the free energy barrier as F(transition state) - F(reactant).
    Reactant and product states are identified from the minima at negative
    and positive reaction coordinate values, respectively.
    The transition state is taken as the maximum located between these minima.

    Parameters
    ----------
    f : ndarray
        Free energy values (in kJ/mol).
    bins : ndarray
        Reaction coordinate values.

    Returns
    -------
    reaction : float
        Reaction free energy (in kJ/mol).
    barrier : float
        Free energy barrier (in kJ/mol).
    """
    f = np.array(f)
    bins = np.array(bins)

    left_mask = (bins < 0) & (~np.isnan(f))
    if not np.any(left_mask):
        return np.nan, np.nan
    left_idx = np.where(left_mask)[0][np.nanargmin(f[left_mask])]
    left_min_bin = bins[left_idx]
    left_min = np.nanmin(f[left_mask])

    right_mask = (bins > 0) & (~np.isnan(f))
    if not np.any(right_mask):
        return np.nan, np.nan
    right_idx = np.where(right_mask)[0][np.nanargmin(f[right_mask])]
    right_min_bin = bins[right_idx]
    right_min = np.nanmin(f[right_mask])

    reaction = right_min - left_min

    ts_mask = (bins > left_min_bin) & (bins < right_min_bin) & (~np.isnan(f))
    if np.any(ts_mask):
        barrier = np.max(f[ts_mask]) - left_min
    else:
        barrier = np.nan
    return reaction, barrier


@pytest.fixture
def profile_errors(free_energy_profiles) -> dict[str, float]:
    """
    Compute the mean absolute error (MAE) of the free energy profiles for each model.

    The MAE is computed for each free energy profile with respect
    to the corresponding reference profile and then averaged over all structures.

    Parameters
    ----------
    free_energy_profiles : dict
        Reference and predicted free energy profiles generated
        by the ``free_energy_profiles`` fixture for all models.

    Returns
    -------
    dict[str, float]
        Average MAE of the free energy profiles for each model.
    """
    results = {}

    f_ref_all = free_energy_profiles["ref"]

    for model in MODELS:
        f_model_all = free_energy_profiles[model]

        if not f_model_all:
            results[model] = None
            continue

        errors = []

        for f_ref, f_model in zip(f_ref_all, f_model_all, strict=True):
            mask = (~np.isnan(f_ref)) & (~np.isnan(f_model))
            if np.any(mask):
                errors.append(mae(f_ref[mask], f_model[mask]))
        if errors:
            results[model] = float(np.mean(errors))
        else:
            results[model] = None

    return results


def get_structures_names():
    """
    Get the names of all available structures from the reference dataset.

    The reference dataset is used to determine the list of structures, which is
    assumed to be the same for all models.

    Returns
    -------
    list[str]
        Names of the available structures without the file extension.
    """
    ref_dir = (
        download_s3_data(
            key="inputs/molecular_reactions/HPHT_CH4_H2O/HPHT_CH4_H2O.zip",
            filename="HPHT_CH4_H2O.zip",
        )
        / "HPHT_CH4_H2O"
    )
    ref_files = sorted(ref_dir.glob("*.data"))
    return [f.stem for f in ref_files]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_reaction_free_energy.json",
    title="Free Energy of Reaction",
    x_label="Predicted ΔF / kJ/mol",
    y_label="Reference ΔF / kJ/mol",
    hoverdata={"Structure": get_structures_names()},
)
def reaction_free_energies(free_energy_profiles) -> dict[str, list[float]]:
    """
    Generate the reaction free energies for all reference and predicted profiles.

    Parameters
    ----------
    free_energy_profiles : dict
        Output of the ``free_energy_profiles`` fixture containing the reference
        and predicted free energy profiles.

    Returns
    -------
    dict[str, list[float]]
        Dictionary containing the reaction free energies for the reference
        profiles and for each model.
    """
    bins = free_energy_profiles["x"]

    results: dict[str, list[float]] = {"ref": []} | {mlip: [] for mlip in MODELS}

    for _i, f_ref in enumerate(free_energy_profiles["ref"]):
        results["ref"].append(reaction_free_energy(f_ref, bins)[0])

    for model in MODELS:
        for _i, f_model in enumerate(free_energy_profiles[model]):
            df = reaction_free_energy(f_model, bins)[0]
            results[model].append(df)

    return results


@pytest.fixture
def reaction_free_energy_errors(reaction_free_energies) -> dict[str, float]:
    """
    Compute the mean absolute error (MAE) of the reaction free energies for each model.

    Parameters
    ----------
    reaction_free_energies : dict
        Output of the ``reaction_free_energies`` fixture containing the reference
        and predicted reaction free energies.

    Returns
    -------
    dict[str, float]
        Mean absolute error of the reaction free energies for each model.
    """
    ref = np.array(reaction_free_energies["ref"])
    mae_values: dict[str, float] = {}
    for model_name in MODELS:
        predictions = np.array(reaction_free_energies[model_name])

        if ref.size == 0 or predictions.size == 0:
            mae_values[model_name] = None
            continue
        mask = (~np.isnan(ref)) & (~np.isnan(predictions))
        if np.any(mask):
            mae_values[model_name] = mae(ref[mask], predictions[mask])
        else:
            mae_values[model_name] = None

    return mae_values


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_barrier_free_energy.json",
    title="Free Energy Barrier",
    x_label="Predicted ΔF# / kJ/mol",
    y_label="Reference ΔF# / kJ/mol",
    hoverdata={"Structure": get_structures_names()},
)
def reaction_barriers(free_energy_profiles) -> dict[str, list[float]]:
    """
    Generate free energy barriers for the reference and predicted free energy profiles.

    Parameters
    ----------
    free_energy_profiles : dict
        Output of the ``free_energy_profiles`` fixture containing the reference
        and predicted free energy profiles.

    Returns
    -------
    dict[str, list[float]]
        Dictionary containing the free energy barriers for the reference profiles
        and for each model.
    """
    bins = free_energy_profiles["x"]

    results: dict[str, list[float]] = {"ref": []} | {mlip: [] for mlip in MODELS}

    for _i, f_ref in enumerate(free_energy_profiles["ref"]):
        results["ref"].append(reaction_free_energy(f_ref, bins)[1])

    for model in MODELS:
        for _i, f_model in enumerate(free_energy_profiles[model]):
            barrier = reaction_free_energy(f_model, bins)[1]
            results[model].append(barrier)

    return results


@pytest.fixture
def reaction_barriers_errors(reaction_barriers) -> dict[str, float]:
    """
    Compute the mean absolute error (MAE) of the free energy barriers for each model.

    Parameters
    ----------
    reaction_barriers : dict
        Output of the ``reaction_barriers`` fixture containing the reference and
        predicted free energy barriers.

    Returns
    -------
    dict[str, float | None]
        Mean absolute error of the free energy barriers for each model. A value of
        ``None`` is returned when no valid comparison can be performed.
    """
    ref = np.array(reaction_barriers["ref"])
    mae_values: dict[str, float] = {}
    for model_name in MODELS:
        predictions = np.array(reaction_barriers[model_name])
        if ref.size == 0 or predictions.size == 0:
            mae_values[model_name] = None
            continue
        mask = (~np.isnan(ref)) & (~np.isnan(predictions))

        if np.any(mask):
            mae_values[model_name] = mae(ref[mask], predictions[mask])
        else:
            mae_values[model_name] = None

    return mae_values


@pytest.fixture
@build_table(
    filename=OUT_PATH / "fes_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    profile_errors: dict[str, float],
    reaction_free_energy_errors: dict[str, float],
    reaction_barriers_errors: dict[str, float],
) -> dict[str, dict]:
    """
    Collect all benchmark metrics into the format expected by ``build_table``.

    Parameters
    ----------
    profile_errors : dict[str, float]
        Mean absolute errors of the free energy profiles.
    reaction_free_energy_errors : dict[str, float]
        Mean absolute errors of the reaction free energies.
    reaction_barriers_errors : dict[str, float]
        Mean absolute errors of the free energy barriers.

    Returns
    -------
    dict[str, dict]
        Dictionary containing all benchmark metrics.
    """
    return {
        "Free Energy Profile MAE": profile_errors,
        "Free Energy of reaction MAE": reaction_free_energy_errors,
        "Free Energy barrier MAE": reaction_barriers_errors,
    }


def test_hpht_ch4_h2o(metrics: dict[str, dict]) -> None:
    """
    Test the HPHT CH4/H2O analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Benchmark metrics used to build the results table.
    """
    write_struct_info(
        data_path=sorted((CALC_PATH / "mock").glob("*.extxyz")),
        out_path=OUT_PATH,
        index=0,
    )
    return
