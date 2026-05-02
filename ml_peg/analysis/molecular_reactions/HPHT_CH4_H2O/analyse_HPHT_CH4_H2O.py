"""Analyse HPHT_CH4_H2O benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_dispersion_name_map,load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models
import numpy as np
from ase.neighborlist import NeighborList, get_connectivity_matrix, neighbor_list, NewPrimitiveNeighborList
from ase.io import iread
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from ml_peg.calcs.utils.utils import download_s3_data


MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "molecular_reactions" / "HPHT_CH4_H2O" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "HPHT_CH4_H2O"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)
OUT_PATH.mkdir(parents=True, exist_ok=True)

def make_fes(data, start, end, n_bins,T):
    """
    Creates a numpy histogram from raw reaction coordinate values,
    then computes free energy profile where the minimum is set to zero together with centered bin values.
    Free energy value is set to NaN if the probability is 0 (avoiding infinite values).

    Parameters
    ----------
        data (list or array) : raw reaction coordinate values (in A)
        start (float) : first bin value
        end (float) : last bin value
        n_bins (int) : total number of desired bins
        T (float) : temperature of the simulation (in K)

    Returns
    -------
        bin_centers (ndarray) : centered bins values (in A)
        F (ndarray) : free energy values (in kJ/mol)
    """
    hist, bin_edges = np.histogram(data, bins=n_bins, range=(start, end))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    prob = hist / np.sum(hist)
    k=8.314
    mask = prob != 0
    F = np.full_like(prob, np.nan)
    F[mask] = -k * T * np.log(prob[mask])
    F -= np.nanmin(F)
    return bin_centers, F

def build_H_heavy_neighborlist_simple(atoms, cutoff_H_to_heavy):
    """
    Compute a neighborlist to search for H atoms in a sphere of radius cutoff_H_to_heavy 
    centered on heavy atoms (C,O)
    It ensures: 
      - H-heavy connection via bothways
      - H-H connection impossible
      - heavy-heavy connection possible but will be ignored in the use of the function
    
    Parameters
    ----------
        atoms : atom object
        cutoff_H_to_heavy : distance cutoff (in A) to use

    Returns
    -------
        nl :  neighborlist
    """
    
    symbols = atoms.get_chemical_symbols()
    cutoffs = []

    for sym in symbols:
        if sym == "H":
            cutoffs.append(0.0)
        elif sym in ("C","O"):
            cutoffs.append(cutoff_H_to_heavy)
        else:
            cutoffs.append(0.0)

    nl = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=True)
    nl.update(atoms)
    return nl

def second_nearest_heavy_outside_molecule(h, atoms, nl, labels):
    """
    Find the nearest heavy atoms for an hydrogen atom, outside of its molecule.
    It corresponds to the second nearest neighbor.
    
    Parameters
    ----------
        h (int) : hydrogen atom index to search for
        atoms : atom object
        nl : the neighbor list used for the search
        labels (list): list of atoms indexes inside each molecule

    Returns
    -------
        heavy2 (int): index of the second nearesr neighbor of hydrogen h
        min_dist (float): distance between h and heavy2
    """
    mol_h = labels[h]

    neighbors, offsets = nl.get_neighbors(h)

    min_dist = float("inf")
    heavy2 = None
    for idx, shift in zip(neighbors, offsets):
        # 🔹 exclure ceux dans la même molécule
        if labels[idx] == mol_h:
            continue

        dr = atoms.positions[h] - (atoms.positions[idx] + shift @ atoms.cell)
        dist = np.linalg.norm(dr)

        if dist < min_dist:
            min_dist = dist
            heavy2 = idx

    return heavy2, min_dist

def build_connectivity_matrix_COH_HH_force(atoms, dCC=1.8, dOO=1.6, dCO=1.7,
                                           dCH=1.4, dOH=1.3, dHH=1.2):
    """
    Build the C/O/H connectivity matrix as followed:
        1. H atoms connected to the nearest heavy atoms with respect to pair cutoff values (dCH,dOH).
        2. Connection between heavy atoms with respect to pair cutoff values (dCC,dCO,dOO).
        3. Connection between H atoms from the H_orphan list which is composed of 
        all the H atoms left without nearest heavy atoms after the step 1. 
        This stands as a search of H2 molecules.
        4. After step 3, the remaining H atoms not connected to anyone go into the H_lonely list.
        From this, they are attached as in step 1 to the nearest heavy atoms without using dCH/dOH cutoff.
    
    Parameters
    ----------
        atoms: atom object
        dCC, dOO, dCO, dCH, dOH, dHH (float): distance based pair cutoffs

    Returns
    -------
        matrix : lil_matrix sparse
        H_to_heavy : dict {H_index: heavy_index}
        H_orphan (list): list of H atoms indexes out of all dCH/dOH radius spheres
        H_lonely (list): list of H atoms from H_orphan forced to be connected to a heavy atom
    """
    n_atoms = len(atoms)
    idx_H = [i for i, a in enumerate(atoms) if a.symbol == 'H']
    idx_heavy = [i for i, a in enumerate(atoms) if a.symbol in ('C','O')]
    pair_cutoffs = {
        ('C','C'): dCC,
        ('O','O'): dOO,
        ('C','O'): dCO,
        ('O','C'): dCO,
        ('C','H'): dCH,
        ('H','C'): dCH,
        ('O','H'): dOH,
        ('H','O'): dOH,
        ('H','H'): dHH,
    }
    matrix = lil_matrix((n_atoms, n_atoms), dtype=int)
    """
    ------------------------------------------------------------------------------
    Search of H atoms contained in dCH/dOH radius spheres centered on heavy atoms.
    ------------------------------------------------------------------------------
    """
    symbols = atoms.get_chemical_symbols()
    cutoffs = []

    for sym in symbols:
        if sym == "H":
            cutoffs.append(0.0)  # H ne détecte rien
        elif sym == "C":
            cutoffs.append(pair_cutoffs.get(('C','H'), 0.0))  # heavy détecte H
        elif sym == "O":
            cutoffs.append(pair_cutoffs.get(('O','H'), 0.0))  # heavy détecte H

    nl_H = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=True, primitive=NewPrimitiveNeighborList)
    nl_H.update(atoms)
    """
    -----------------------------------------------------------------------------------------
    Connection of each H atoms contained in dCH/dOH radius spheres with its nearest neighbor.
    Other H atoms are sent to the H_oprhan list.
    -----------------------------------------------------------------------------------------
    """
    H_to_heavy = {}
    H_orphan = []
    for h in idx_H:
        neighbors, offsets = nl_H.get_neighbors(h)
        heavy_neighbors = []
        min_dist = float("inf")
        nearest = None
        if len(neighbors)==0:
            H_orphan.append(h)
        else:
            for idx, shift in zip(neighbors, offsets):
                dr = atoms.positions[h] - (atoms.positions[idx] + shift @ atoms.cell)
                dist = np.linalg.norm(dr)

                if dist < min_dist:
                    min_dist = dist
                    nearest = idx
            H_to_heavy[h]=nearest

    for h, heavy in H_to_heavy.items():
        matrix[h, heavy] = 1
        matrix[heavy, h] = 1

    """
    ---------------------------------------------------------------------------------------------------
    Manual connection between heavy atoms with repsect to heavy pair cutoffs from a global neighbor list. 
    ---------------------------------------------------------------------------------------------------
    """
    max_cutoff = max(pair_cutoffs.values())
    i_list, j_list, d_list = neighbor_list('ijd', atoms, cutoff=max_cutoff)
    symbols = atoms.get_chemical_symbols()

    for idx in range(len(i_list)):
        i = i_list[idx]
        j = j_list[idx]

        sym_i = symbols[i]
        sym_j = symbols[j]

        if sym_i not in ('C','O') or sym_j not in ('C','O'):
            continue

        pair = tuple(sorted((sym_i, sym_j)))
        allowed = pair_cutoffs.get(pair, 0.0)

        if d_list[idx] <= allowed:
            matrix[i, j] = 1
            matrix[j, i] = 1

    """
    --------------------------------------------------------------------
    Connection between orphan H atoms with respect to dHH cutoff.
    Remaining unconnected orphan H atoms are sent to the H_lonely list.
    --------------------------------------------------------------------
    """
    H_lonely = []
    if H_orphan:
        cutoffs_HH = np.zeros(n_atoms)
        for h in H_orphan:
            cutoffs_HH[h] = dHH
        nl_HH = NeighborList(cutoffs_HH, skin=0, self_interaction=False, bothways=True)
        nl_HH.update(atoms)
        for h in H_orphan:
            neighbors, offsets = nl_HH.get_neighbors(h)
            has_H_neighbor = False
            for i, n in enumerate(neighbors):
                if n in H_orphan:
                    shift = offsets[i] @ atoms.cell
                    dr = atoms.positions[h] - (atoms.positions[n] + shift)
                    if np.linalg.norm(dr) <= dHH:
                        matrix[h, n] = 1
                        matrix[n, h] = 1
                        has_H_neighbor = True
            if not has_H_neighbor:
                H_lonely.append(h)
    """
    -----------------------------------------------------------------------------
    Manual connection of lonely H atoms to the nearest heavy atom without cutoff.
    Return of matrix, H_to_heavy dictionnary, and H_orphan and H_lonely lists.
    -----------------------------------------------------------------------------
    """
    for h in H_lonely:
        min_dist = float('inf')
        nearest_heavy = None
        for n in idx_heavy:
            dist = atoms.get_distance(h, n, mic=True)
            if dist < min_dist:
                min_dist = dist
                nearest_heavy = n
        if nearest_heavy is not None:
            matrix[h, nearest_heavy] = 1
            matrix[nearest_heavy, h] = 1

            H_to_heavy[h] = nearest_heavy

    return matrix, H_to_heavy, H_orphan, H_lonely

def compute_fes(extxyz_file,T):
    """
    Perform a molecular recognition from the build_connectivity_matrix function
    and get the free energy profile of proton hopping between H3O+ ions and CH4 molecules.
    
    Parameters
    ----------
        extxyz_file: input trajectory file obtained from the calculation part
        T: temperature of the simulation (in K)
    
    Return
    ------
        bins: array of reaction coordinate values
        F: array (size of bins) containing associated free energy values
    """
    cutoff_H_to_heavy = 3.0
    frame_max=100000
    coord=[]
    for frame_idx, atoms in enumerate(iread(extxyz_file, format="extxyz")):
        if frame_idx==frame_max:
            break
        matrix, H_to_heavy, H_orphan, H_lonely = build_connectivity_matrix_COH_HH_force(atoms)
        n_mol, labels = connected_components(matrix)
        molecules = []
        for mol_idx in range(n_mol):
            atom_indices = tuple(int(i) for i in np.where(labels == mol_idx)[0])
            counts = {"C": 0, "O": 0, "H": 0}
            for i in atom_indices:
                sym = atoms[i].symbol
                if sym in counts:
                    counts[sym] += 1
            molecules.append({
                "atoms": atom_indices,
                "nC": counts["C"],
                "nH": counts["H"],
                "nO": counts["O"],
                "label": f"C{counts['C']}H{counts['H']}O{counts['O']}"
            })
        
        nl = build_H_heavy_neighborlist_simple(atoms, cutoff_H_to_heavy)

        for h in H_to_heavy:
            h_mol_idx=labels[h]
            if molecules[h_mol_idx]['label'] == "C0H3O1":
                heavy2, dist_heavy2 = second_nearest_heavy_outside_molecule(h, atoms, nl, labels)
                if molecules[labels[heavy2]]['label'] == "C1H4O0":
                    dist_heavy = atoms.get_distance(h, H_to_heavy[h], mic=True)
                    coord.append(dist_heavy-dist_heavy2)
            if molecules[h_mol_idx]['label'] == "C1H5O0":
                heavy2, dist_heavy2 = second_nearest_heavy_outside_molecule(h, atoms, nl, labels)
                if molecules[labels[heavy2]]['label'] == "C0H2O1":
                    dist_heavy = atoms.get_distance(h, H_to_heavy[h], mic=True)
                    coord.append(dist_heavy2-dist_heavy)

    bins, F = make_fes(coord, start=-1.5, end=1.5, n_bins=200, T=T)
    return bins, F

def load_reference_fes(structure_name):
    """
    Load the reference free energy profile of the desired structure.

    Parameters
    ----------
        structure_name: name of the referene structure

    Return
    ------
        bins: array of reaction coordinate values
        F: array (size of bins) containing associated free energy values
    """
    ref_dir= (
        download_s3_data(
            key="inputs/molecular_reactions/HPHT_CH4_H2O/HPHT_CH4_H2O.zip",
            filename="HPHT_CH4_H2O_data.zip"
        ) / "HPHT_CH4_H2O_data"
    )
    ref_file = ref_dir / f"{structure_name}.data"

    if not ref_file.exists():
        raise FileNotFoundError(f"Missing reference file {ref_file}")

    data = np.loadtxt(ref_file)

    bins = data[:,0]
    F = data[:,1]

    return bins, F

@pytest.fixture
def free_energy_profiles():
    """
    Get free energy profile for all CH4/H2O systems for all MODELS.
    Write free energy profile in the data path of the application to be plot by the app.py script

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted free energy profile
    """
    results = {
        "x": None,
        "ref": [],
    } | {model: [] for model in MODELS}
    structures_processed = set()
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue
        extxyz_files = sorted(model_dir.glob("*.extxyz"))
        for xyz_file in extxyz_files:
            structure_name = xyz_file.stem
            bins, F_model = compute_fes(xyz_file,3000)
            save_path = OUT_PATH / model_name / f"{structure_name}.data"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(save_path, np.column_stack((bins,F_model)))
            results[model_name].append(F_model)
            if results["x"] is None:
                results["x"] = bins
            if structure_name not in structures_processed:
                ref_bins, ref_F = load_reference_fes(structure_name)
                save_ref_path = OUT_PATH / f"{structure_name}.data"
                np.savetxt(save_ref_path, np.column_stack((ref_bins,ref_F)))
                results["ref"].append(ref_F)
                structures_processed.add(structure_name)
    return results

def reaction_free_energy(F, bins):
    """
    Compute free energy of reaction (F(product) - F(reactant)) and free energy barrier (F(transition state) - F(reactant)) while avoiding NaN values.
    Search for reactant and product minima (reaction coordinate < 0 and > 0)
    Search for a maximum between reactant and product minima
    
    Parameters
    ----------
        F: array of free energy values
        bins: array of reaction coordinate values

    Return
    ------
        reaction(float): reaction free energy value
        barrier(float): free energy barrier value
    """
    F = np.array(F)
    bins = np.array(bins)
    
    left_mask = (bins < 0) & (~np.isnan(F))
    if np.any(left_mask):
        left_idx = np.where(left_mask)[0][np.nanargmin(F[left_mask])]
        left_min_bin = bins[left_idx]
        left_min = np.nanmin(F[left_mask])
    else:
        left_min_bin = 0
        left_min = 0

    right_mask = (bins > 0) & (~np.isnan(F))
    if np.any(right_mask):
        right_idx = np.where(right_mask)[0][np.nanargmin(F[right_mask])]
        right_min_bin = bins[right_idx]
        right_min = np.nanmin(F[right_mask])
    else:
        right_min_bin = 0
        right_min = 0
    reaction = right_min - left_min

    TS_mask = (bins > left_min_bin) & (bins < right_min_bin) & (~np.isnan(F))
    if np.any(TS_mask):
        barrier = np.max(F[TS_mask]) - left_min
    else:
        barrier = 0
    return reaction, barrier

@pytest.fixture
def profile_errors(free_energy_profiles) -> dict[str, float]:
    """
    METRIC 1: Compute the average value of the mean absolute error on the whole free energy profile 
    with respect to reference ones for each model. 
    (mae running on the bins and then average running on structures)

    Parameters
    ----------
        free_energy_profiles (dict[str, list[array]]): dictionnary containing 
        the list of reference free energy profiles and a list of predicted profiles per model.

    Return
    ------
        results (dict[str, float]): dictionnary containing the average mae 
        done on the reference profile per model
    """
    results = {}

    F_ref_all = free_energy_profiles["ref"]

    for model in MODELS:

        F_model_all = free_energy_profiles[model]

        if not F_model_all:
            results[model] = None
            continue

        errors = []

        for F_ref, F_model in zip(F_ref_all, F_model_all):
            mask = (~np.isnan(F_ref)) & (~np.isnan(F_model))
            if np.any(mask):
                err = mae(F_ref[mask], F_model[mask])
            else:
                error = np.nan
            errors.append(err)

        results[model] = float(np.mean(errors))
    
    return results

def get_structures_names():
    """
    Get the list of structures names from the calculation folder of the first model.
    All models should have results for all the same structures.
    """
    model_name = MODELS[0]
    path = CALC_PATH / model_name
    ref_files = sorted(path.glob("*.extxyz"))
    structures = [f.stem for f in ref_files]
    return structures

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
    Get the reaction free energy for all reference and predicted profiles.

    Returns
    -------
        results (dict[str, list[float]])
    """
    bins = free_energy_profiles["x"]

    results: dict[str, list[float]] = {"ref": []} | {mlip: [] for mlip in MODELS}

    for i, F_ref in enumerate(free_energy_profiles["ref"]):
        results["ref"].append(reaction_free_energy(F_ref, bins)[0])

    for model in MODELS:
        for i, F_model in enumerate(free_energy_profiles[model]):
            dF = reaction_free_energy(F_model, bins)[0]
            results[model].append(dF)

    return results

@pytest.fixture
def reaction_free_energy_errors(reaction_free_energies) -> dict[str, float]:
    """
    METRIC 2: Compute the mean absolute error on the reaction free energy with respect to reference ones.

    Returns
    -------
        mae_values (dict[str, float])
    """
    ref = reaction_free_energies["ref"]
    mae_values: dict[str, float] = {}
    for model_name in MODELS:
        predictions = reaction_free_energies[model_name]
        if ref and predictions:
            mae_values[model_name] = mae(ref, predictions)
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
    Get the free energy barrier for all reference and predicted profiles.

    Returns
    -------
        results (dict[str, list[float]])
    """
    bins = free_energy_profiles["x"]

    results: dict[str, list[float]] = {"ref": []} | {mlip: [] for mlip in MODELS}

    for i, F_ref in enumerate(free_energy_profiles["ref"]):
        results["ref"].append(reaction_free_energy(F_ref, bins)[1])

    for model in MODELS:
        for i, F_model in enumerate(free_energy_profiles[model]):
            barrier = reaction_free_energy(F_model, bins)[1]
            results[model].append(barrier)

    return results

@pytest.fixture
def reaction_barriers_errors(reaction_barriers) -> dict[str, float]:
    """
    METRIC 3: Compute the mean absolute error on the free energy barrier with respect to reference ones.

    Returns
    -------
        mae_values (dict[str, float])
    """
    ref = reaction_barriers["ref"]
    mae_values: dict[str, float] = {}
    for model_name in MODELS:
        predictions = reaction_barriers[model_name]
        if ref and predictions:
            mae_values[model_name] = mae(ref, predictions)
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
    return {
        "FEP_MAE": profile_errors,
        "DF_MAE": reaction_free_energy_errors,
        "DF#_MAE": reaction_barriers_errors,
    }

def test_HPHT_CH4_H2O(metrics: dict[str, dict]) -> None:
    return
