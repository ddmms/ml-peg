"""
Shared analysis helpers for the aqueous/interface MD benchmarks.

The bulk_water, ice and copper_water_interface benchmarks all derive the same
RDF/VDOS/VACF observables from MD trajectories and score them the same way. This
module holds that common logic as plain functions parametrised by the per-benchmark
paths and model list, so each ``analyse_*.py`` reduces to thin pytest fixtures that
delegate here. The low-level primitives live in
:mod:`ml_peg.analysis.utils.aml_md_analysis`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import pickle

from ase.io import read
import mdtraj as mdt
import numpy as np

from ml_peg.analysis.utils import aml_md_analysis as aml

# ASE writes/reads positions and velocities in Å; mdtraj stores them in nm.
ANG_TO_NM = 0.1


def _load_positions(traj_file: Path, topology: str) -> mdt.Trajectory:
    """
    Build an mdtraj position trajectory from a janus-core ``-traj.extxyz`` file.

    Reproduces what ``aml_md_analysis.load_with_cell`` produced from the legacy
    ``md-pos.xyz`` (positions in nm, unit cell injected from the topology PDB),
    so the shipped reference data remains valid.

    Parameters
    ----------
    traj_file
        Path to the janus-core trajectory (positions + momenta).
    topology
        Path to the PDB providing topology and cell information.

    Returns
    -------
    mdtraj.Trajectory
        Position trajectory with cell information injected.
    """
    frames = read(traj_file, ":")
    top_frame = mdt.load_frame(str(topology), 0)
    xyz = np.array([atoms.get_positions() for atoms in frames]) * ANG_TO_NM
    trj = mdt.Trajectory(xyz, top_frame.topology)
    n_frames = len(trj)
    trj.unitcell_lengths = top_frame.unitcell_lengths.repeat(n_frames, axis=0)
    trj.unitcell_angles = top_frame.unitcell_angles.repeat(n_frames, axis=0)
    return trj


def _load_velocities(traj_file: Path, topology: str) -> mdt.Trajectory:
    """
    Build an mdtraj velocity trajectory from a janus-core ``-traj.extxyz`` file.

    Reproduces what ``mdtraj.load`` produced from the legacy ``md-velc.xyz`` (ASE
    velocities scaled Å→nm, unit cell injected from the topology PDB), so VDOS/VACF
    remain comparable to the shipped reference data. Velocities are recovered from
    the trajectory momenta via ``atoms.get_velocities()`` (masses in the file, so
    deuterated systems are handled correctly).

    Parameters
    ----------
    traj_file
        Path to the janus-core trajectory (positions + momenta + masses).
    topology
        Path to the PDB providing topology and cell information.

    Returns
    -------
    mdtraj.Trajectory
        Velocity "trajectory" with cell information injected.
    """
    frames = read(traj_file, ":")
    top_pdb = mdt.load_pdb(str(topology))
    vel = np.array([atoms.get_velocities() for atoms in frames]) * ANG_TO_NM
    trj = mdt.Trajectory(vel, top_pdb.topology)
    n_frames = len(trj)
    trj.unitcell_lengths = np.repeat(top_pdb.unitcell_lengths, n_frames, axis=0)
    trj.unitcell_vectors = np.repeat(top_pdb.unitcell_vectors, n_frames, axis=0)
    return trj


def get_rdf_keys(data_path: Path) -> list:
    """
    Return list of relevant RDF key names.

    Parameters
    ----------
    data_path
        Path to the benchmark input data (containing ``init.xyz``).

    Returns
    -------
    list
        List of all RDF keys.
    """
    rdf_keys = []
    xyz_file = data_path / "init.xyz"
    atoms = read(xyz_file)
    all_species = set(atoms.get_chemical_symbols())
    all_species = sorted(all_species)
    for species1 in all_species:
        for species2 in all_species:
            pair = "-".join(sorted([species1, species2]))
            if pair not in rdf_keys:
                rdf_keys.append(pair)
    return rdf_keys


def get_elements(data_path: Path) -> list:
    """
    Return list of relevant element names.

    Parameters
    ----------
    data_path
        Path to the benchmark input data (containing ``init.xyz``).

    Returns
    -------
    list
        List of all elements.
    """
    elements = set()
    xyz_file = data_path / "init.xyz"
    atoms = read(xyz_file)
    for symbol in atoms.get_chemical_symbols():
        elements.add(symbol)
    return sorted(elements)


def create_rdfs(
    models: list, data_path: Path, calc_path: Path, curve_path: Path
) -> dict[str, dict]:
    """
    Create RDFs for all models.

    Parameters
    ----------
    models
        Names of MLIP models to analyse.
    data_path
        Path to the benchmark input data.
    calc_path
        Path to the calculation outputs (one subdir per model).
    curve_path
        Path to write per-model RDF curve pickles for app use.

    Returns
    -------
    dict[str, dict]
        Dictionary of RDFs for all models.
    """
    rdfs = {"ref": None} | dict.fromkeys(models)

    # Load reference RDF
    with open(f"{data_path}/rdf_reference.pkl", "rb") as f_in:
        rdfs["ref"] = pickle.load(f_in)

    # Load topology
    ref_topology = f"{data_path}/init.pdb"

    # Check if rdf curves directory exists, if not create it
    if not curve_path.exists():
        curve_path.mkdir(parents=True, exist_ok=True)

    # Load model RDFs
    for model_name in models:
        model_dir = calc_path / model_name

        if not model_dir.exists():
            continue

        # Check if rdf curves directory exists for model, if not create it
        if not (curve_path / model_name).exists():
            (curve_path / model_name).mkdir(parents=True, exist_ok=True)

        traj_file = model_dir / "md-traj.extxyz"
        if not traj_file.exists():
            continue

        test_trj = _load_positions(traj_file, ref_topology)

        rdfs[model_name] = aml.compute_all_rdfs(test_trj)

        # Make sure keys are in alphabetical order
        rdfs_keys = list(rdfs[model_name].keys())
        for key in rdfs_keys:
            split_key = key.split("-")
            split_key.sort()
            altered_key = "-".join(split_key)
            if altered_key != key:
                rdfs[model_name][altered_key] = rdfs[model_name].pop(key)

        # Write rdf curves to file for app use
        with open(curve_path / model_name / "rdf_curves.pkl", "wb") as f_out:
            pickle.dump(rdfs[model_name], f_out)

    return rdfs


def create_vdos(
    models: list, data_path: Path, calc_path: Path, curve_path: Path
) -> dict[str, dict]:
    """
    Create VDOS for all models.

    Parameters
    ----------
    models
        Names of MLIP models to analyse.
    data_path
        Path to the benchmark input data.
    calc_path
        Path to the calculation outputs (one subdir per model).
    curve_path
        Path to write per-model VDOS curve pickles for app use.

    Returns
    -------
    dict[str, dict]
        Dictionary of VDOS for all models.
    """
    vdos = {"ref": None} | dict.fromkeys(models)

    # Load reference RDF
    with open(f"{data_path}/vdos_reference.pkl", "rb") as f_in:
        vdos["ref"] = pickle.load(f_in)

    # Load topology
    ref_topology = f"{data_path}/init.pdb"

    # Check if vdos curves directory exists, if not create it
    if not curve_path.exists():
        curve_path.mkdir(parents=True, exist_ok=True)

    # Load model RDFs
    for model_name in models:
        model_dir = calc_path / model_name

        if not model_dir.exists():
            continue

        # Check if vdos curves directory exists for model, if not create it
        if not (curve_path / model_name).exists():
            (curve_path / model_name).mkdir(parents=True, exist_ok=True)

        traj_file = model_dir / "md-traj.extxyz"
        if not traj_file.exists():
            continue

        test_vel = _load_velocities(traj_file, ref_topology)

        ref_dt = 1

        vdos[model_name] = aml.compute_all_vdos(test_vel, ref_dt)

        # Write rdf curves to file for app use
        with open(curve_path / model_name / "vdos_curves.pkl", "wb") as f_out:
            pickle.dump(vdos[model_name], f_out)

    return vdos


def create_vacf(
    models: list,
    data_path: Path,
    calc_path: Path,
    curve_path: Path,
    ref_vel_path: Path,
) -> dict[str, dict]:
    """
    Create VACF for all models.

    Parameters
    ----------
    models
        Names of MLIP models to analyse.
    data_path
        Path to the benchmark input data.
    calc_path
        Path to the calculation outputs (one subdir per model).
    curve_path
        Path to write per-model VACF curve pickles for app use.
    ref_vel_path
        Path to the reference velocity trajectory.

    Returns
    -------
    dict[str, dict]
        Dictionary of VACF for all models.
    """
    vacf = {"ref": None} | dict.fromkeys(models)

    # Load reference RDF
    with open(f"{data_path}/vacf_reference.pkl", "rb") as f_in:
        vacf["ref"] = pickle.load(f_in)

    # Load topology
    ref_topology = f"{data_path}/init.pdb"

    # Check if vacf curves directory exists, if not create it
    if not curve_path.exists():
        curve_path.mkdir(parents=True, exist_ok=True)

    # Load model RDFs
    for model_name in models:
        model_dir = calc_path / model_name

        if not model_dir.exists():
            continue

        # Check if vacf curves directory exists for model, if not create it
        if not (curve_path / model_name).exists():
            (curve_path / model_name).mkdir(parents=True, exist_ok=True)

        traj_file = model_dir / "md-traj.extxyz"
        if not traj_file.exists():
            continue

        test_vel = _load_velocities(traj_file, ref_topology)
        # Topology used below to inject cell info into the reference velocities
        test_top = mdt.load_pdb(ref_topology)

        ref_dt = 1

        # Check if length of trajectory matches reference
        ref_length = len(list(vacf["ref"].values())[0][0])
        test_length = len(test_vel)
        min_length = min(test_length, ref_length)

        if vacf["ref"] is not None:
            if test_length != ref_length:
                # Truncate both to shortest length - and recalculate reference if needed

                test_vel = test_vel[test_length - min_length :]
                if len(test_vel) != ref_length:
                    ref_vel = mdt.load(ref_vel_path, top=ref_topology, stride=1)
                    ref_vel.unitcell_lengths = np.repeat(
                        test_top.unitcell_lengths, len(ref_vel), axis=0
                    )
                    ref_vel.unitcell_vectors = np.repeat(
                        test_top.unitcell_vectors, len(ref_vel), axis=0
                    )
                    ref_vel_trunc = ref_vel[ref_length - min_length :]
                    vacf_ref_trunc = aml.compute_all_vacfs(ref_vel_trunc, ref_dt)
                    # Write updated reference vacf curves to file for app use

        else:
            vacf_ref_trunc = vacf["ref"]

        vacf[f"ref_{model_name}"] = vacf_ref_trunc
        vacf[model_name] = aml.compute_all_vacfs(test_vel, ref_dt)

        # Write rdf curves to file for app use
        with open(curve_path / model_name / "vacf_curves.pkl", "wb") as f_out:
            pickle.dump(vacf[model_name], f_out)

    return vacf


def property_scores(
    created: dict[str, dict],
    models: list,
    *,
    errors_fn: Callable,
    ref_key: Callable,
) -> dict[str, list]:
    """
    Get per-pair scores for a property (RDF, VDOS or VACF) for all models.

    Parameters
    ----------
    created
        Created property data for all models (as returned by the ``create_*``
        helpers).
    models
        Names of MLIP models to analyse.
    errors_fn
        Function computing per-pair errors between reference and model data
        (e.g. ``aml.compute_all_errors`` or ``aml.compute_all_errors_vacf``).
    ref_key
        Callable mapping a model name to the key of its reference data in
        ``created`` (``lambda m: "ref"`` for RDF/VDOS, ``lambda m: f"ref_{m}"``
        for VACF).

    Returns
    -------
    dict[str, list]
        Dictionary of per-pair scores for all models.
    """
    results = {"ref": []} | dict.fromkeys(models)
    store_ref = False
    for model_name in models:
        if created[model_name] is None:
            continue
        errors = errors_fn(created[ref_key(model_name)], created[model_name])
        # Mae is stored per pair in [2] slot of tuple
        scores = [aml.error_score_percentage(error[2]) for error in errors.values()]

        results[model_name] = scores
        if not store_ref:
            results["ref"] = [100.0 for _ in scores]

        store_ref = True

    return results


def mean_score(scores: dict[str, list], models: list) -> dict[str, float]:
    """
    Get mean score for all models.

    Parameters
    ----------
    scores
        Per-pair scores for all models (as returned by :func:`property_scores`).
    models
        Names of MLIP models to analyse.

    Returns
    -------
    dict[str, float]
        Dictionary of mean scores for all models.
    """
    mean_results = {"ref": 100.0} | dict.fromkeys(models)
    for model_name, model_scores in scores.items():
        if model_scores is None:
            continue
        mean_results[model_name] = np.mean(model_scores)
    return mean_results


def build_bar_data(
    scores: dict[str, list],
    created: dict[str, dict],
    models: list,
    *,
    metric_key: str,
    metric_label: str,
    ylabel: str,
    xlabel: str,
    errors_fn: Callable,
    ref_key: Callable,
    xlim: list | None = None,
) -> dict:
    """
    Build interactive data structure for a property bar plot.

    Parameters
    ----------
    scores
        Per-pair scores for all models (as returned by :func:`property_scores`).
    created
        Created property data for all models.
    models
        Names of MLIP models to analyse.
    metric_key
        Key identifying the metric (e.g. ``"rdf_score"``).
    metric_label
        Human-readable label for the metric (e.g. ``"RDF Score"``).
    ylabel
        Label for the per-pair curve y-axis.
    xlabel
        Label for the per-pair curve x-axis.
    errors_fn
        Function computing per-pair errors between reference and model data.
    ref_key
        Callable mapping a model name to the key of its reference data in
        ``created``.
    xlim
        Optional x-axis limits for the per-pair curve. Default is None.

    Returns
    -------
    dict
        Interactive data structure for the property bar plot.
    """
    # Get all unique pairs from the reference data
    ref_pairs = list(created["ref"].keys()) if created["ref"] else []

    models_data = {}

    for model_name in models:
        if scores.get(model_name) is None or created.get(model_name) is None:
            continue

        model_data = {}
        ref_data = created[ref_key(model_name)]
        model_prop_data = created[model_name]

        # Get scores for each pair
        prop_errors = errors_fn(ref_data, model_prop_data)

        model_data["metrics"] = {}
        model_data["metrics"][metric_key] = {}
        pair_point_list = []
        for pair in ref_pairs:
            if pair in prop_errors:
                error_data = prop_errors[pair]
                d_score = aml.error_score_percentage(error_data[2])  # MAE score

                point_data = {
                    "x_values": error_data[0].tolist(),  # r values
                    "ref": ref_data[pair][1].tolist(),  # reference curve
                    "pred": model_prop_data[pair][1].tolist(),  # predicted curve
                    "error": error_data[1].tolist(),
                    "ylabel": ylabel,
                    "xlabel": xlabel,  # error array
                }
                if xlim is not None:
                    point_data["xlim"] = xlim

                pair_point_list.append(
                    {
                        "label": pair,
                        "value": d_score,
                        "data": point_data,
                    }
                )
        model_data["metrics"][metric_key] = pair_point_list
        models_data[model_name] = model_data

    # Create metrics mapping
    metrics = {metric_key: metric_label}

    return {"models": models_data, "metrics": metrics, "pairs": ref_pairs}
