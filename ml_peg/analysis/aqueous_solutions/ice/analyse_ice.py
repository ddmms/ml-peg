"""Analysis of ice benchmark."""

from __future__ import annotations

from pathlib import Path
import pickle

from ase.io import read
import mdtraj as mdt
import numpy as np
import pytest

from ml_peg.analysis.aqueous_solutions.ice import aml
from ml_peg.analysis.aqueous_solutions.ice.decorators import (
    cell_to_bar,
)
from ml_peg.analysis.utils.decorators import (
    build_table,
)
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "aqueous_solutions" / "ice" / "outputs"
OUT_PATH = APP_ROOT / "data" / "aqueous_solutions" / "ice"
RDF_CURVE_PATH = OUT_PATH / "rdf_curves"
VDOS_CURVE_PATH = OUT_PATH / "vdos_curves"
VACF_CURVE_PATH = OUT_PATH / "vacf_curves"


METRIC_LABELS = {
    "rdf_score": "RDF Score",
    "vdos_score": "VDOS Score",
    "vacf_score": "VACF Score",
}
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATA_PATH = (
    download_s3_data(
        filename="ice.zip",
        key="inputs/aqueous_solutions/ice/ice.zip",
    )
    / "ice"
)
REF_VEL_PATH = DATA_PATH / "pbe-d3-md-vel.xyz"


def get_rdf_keys() -> list:
    """
    Return list of relevant RDF key names.

    Returns
    -------
    list
        List of all RDF keys.
    """
    rdf_keys = []
    xyz_file = DATA_PATH / "init.xyz"
    atoms = read(xyz_file)
    all_species = set(atoms.get_chemical_symbols())
    all_species = sorted(all_species)
    for species1 in all_species:
        for species2 in all_species:
            pair = "-".join(sorted([species1, species2]))
            if pair not in rdf_keys:
                rdf_keys.append(pair)
    return rdf_keys


def get_elements() -> list:
    """
    Return list of relevant element names.

    Returns
    -------
    list
        List of all elements.
    """
    elements = set()
    xyz_file = DATA_PATH / "init.xyz"
    atoms = read(xyz_file)
    for symbol in atoms.get_chemical_symbols():
        elements.add(symbol)
    return sorted(elements)


@pytest.fixture
def created_rdfs() -> dict[str, dict]:
    """
    Create RDFs for all models.

    Returns
    -------
    dict[str, dict]
        Dictionary of RDFs for all models.
    """
    rdfs = {"ref": None} | dict.fromkeys(MODELS)

    # Load reference RDF
    with open(f"{DATA_PATH}/rdf_reference.pkl", "rb") as f_in:
        rdfs["ref"] = pickle.load(f_in)

    # Load topology
    ref_topology = f"{DATA_PATH}/init.pdb"

    # Check if rdf curves directory exists, if not create it
    if not RDF_CURVE_PATH.exists():
        RDF_CURVE_PATH.mkdir(parents=True, exist_ok=True)

    # Load model RDFs
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        # Check if rdf curves directory exists for model, if not create it
        if not (RDF_CURVE_PATH / model_name).exists():
            (RDF_CURVE_PATH / model_name).mkdir(parents=True, exist_ok=True)

        position_xyz = model_dir / "md-pos.xyz"
        if not position_xyz.exists():
            continue

        test_trj = aml.load_with_cell(position_xyz, top=ref_topology)

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
        with open(RDF_CURVE_PATH / model_name / "rdf_curves.pkl", "wb") as f_out:
            pickle.dump(rdfs[model_name], f_out)

    return rdfs


def plot_rdfs(model: str, element_pair: str, rdf: dict, error: bool = False) -> None:
    """
    Plot RDF paths and save all structure files.

    Parameters
    ----------
    model
        Name of MLIP.
    element_pair
        Element pair for RDF.
    rdf
        RDF data.
    error
        Whether to plot error or raw RDF.
    """

    def plot_rdf() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot a RDF and save the structure file.

        Returns
        -------
        dict[str, tuple[list[float], list[float]]]
            Dictionary of tuples of image/energy for each model.
        """
        results = {}
        results[model] = [
            rdf[0].tolist(),
            rdf[1].tolist(),
        ]

        return results

    plot_rdf()


@pytest.fixture
def rdf_scores(created_rdfs: dict[str, dict]) -> dict[str, float]:
    """
    Get Average RDF score for all models.

    Parameters
    ----------
    created_rdfs
        Created RDFs for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Average RDF scores for all models.
    """
    results = {"ref": []} | dict.fromkeys(MODELS)
    store_ref = False
    for model_name in MODELS:
        if created_rdfs[model_name] is None:
            continue
        rdf_errors = aml.compute_all_errors(
            created_rdfs["ref"], created_rdfs[model_name]
        )
        # Mae is stored per pair in [2] slot of tuple
        rdf_scores = [
            aml.error_score_percentage(error[2]) for error in rdf_errors.values()
        ]

        results[model_name] = rdf_scores
        if not store_ref:
            results["ref"] = [100.0 for _ in rdf_scores]

        store_ref = True

    return results


@pytest.fixture
def mean_rdf_score(rdf_scores: dict[str, float]) -> dict[str, float]:
    """
    Get Mean RDF score for all models.

    Parameters
    ----------
    rdf_scores
        Average RDF scores for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Mean RDF scores for all models.
    """
    mean_results = {"ref": 100.0} | dict.fromkeys(MODELS)
    for model_name, scores in rdf_scores.items():
        if scores is None:
            continue
        mean_results[model_name] = np.mean(scores)
    return mean_results


# -------- VDOS


@pytest.fixture
def created_vdos() -> dict[str, dict]:
    """
    Create VDOS for all models.

    Returns
    -------
    dict[str, dict]
        Dictionary of VDOS for all models.
    """
    vdos = {"ref": None} | dict.fromkeys(MODELS)

    # Load reference RDF
    with open(f"{DATA_PATH}/vdos_reference.pkl", "rb") as f_in:
        vdos["ref"] = pickle.load(f_in)

    # Load topology
    ref_topology = f"{DATA_PATH}/init.pdb"

    # Check if vdos curves directory exists, if not create it
    if not VDOS_CURVE_PATH.exists():
        VDOS_CURVE_PATH.mkdir(parents=True, exist_ok=True)

    # Load model RDFs
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        # Check if vdos curves directory exists for model, if not create it
        if not (VDOS_CURVE_PATH / model_name).exists():
            (VDOS_CURVE_PATH / model_name).mkdir(parents=True, exist_ok=True)

        vel_xyz = model_dir / "md-velc.xyz"
        if not vel_xyz.exists():
            continue

        test_vel = mdt.load(vel_xyz, top=ref_topology, stride=1)
        test_top = mdt.load_pdb(ref_topology)

        test_vel.unitcell_lengths = np.repeat(
            test_top.unitcell_lengths, len(test_vel), axis=0
        )
        test_vel.unitcell_vectors = np.repeat(
            test_top.unitcell_vectors, len(test_vel), axis=0
        )

        ref_dt = 1

        vdos[model_name] = aml.compute_all_vdos(test_vel, ref_dt)

        # Write rdf curves to file for app use
        with open(VDOS_CURVE_PATH / model_name / "vdos_curves.pkl", "wb") as f_out:
            pickle.dump(vdos[model_name], f_out)

    return vdos


@pytest.fixture
def vdos_scores(created_vdos: dict[str, dict]) -> dict[str, float]:
    """
    Get Average VDOS score for all models.

    Parameters
    ----------
    created_vdos
        Created VDOS for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Average VDOS scores for all models.
    """
    results = {"ref": []} | dict.fromkeys(MODELS)
    store_ref = False
    for model_name in MODELS:
        if created_vdos[model_name] is None:
            continue
        vdos_errors = aml.compute_all_errors(
            created_vdos["ref"], created_vdos[model_name]
        )
        # Mae is stored per pair in [2] slot of tuple
        vdos_scores = [
            aml.error_score_percentage(error[2]) for error in vdos_errors.values()
        ]
        results[model_name] = vdos_scores
        if not store_ref:
            results["ref"] = [100.0 for _ in vdos_scores]

        store_ref = True

    return results


@pytest.fixture
def mean_vdos_score(vdos_scores: dict[str, float]) -> dict[str, float]:
    """
    Get Mean VDOS score for all models.

    Parameters
    ----------
    vdos_scores
        Average VDOS scores for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Mean VDOS scores for all models.
    """
    mean_results = {"ref": 100.0} | dict.fromkeys(MODELS)
    for model_name, scores in vdos_scores.items():
        if scores is None:
            continue
        mean_results[model_name] = np.mean(scores)
    return mean_results


@pytest.fixture
@cell_to_bar(
    filename=OUT_PATH / "rdf_score_bar.json",
    x_label="X Coord",
    y_label="RDF Score",
)
def build_rdf_interactive_data(
    rdf_scores: dict[str, list], created_rdfs: dict[str, dict]
) -> dict:
    """
    Build interactive data structure for RDF bar plot.

    Parameters
    ----------
    rdf_scores
        Average RDF scores for all models.
    created_rdfs
        Created RDFs for all models.

    Returns
    -------
    dict
        Interactive data structure for RDF bar plot.
    """
    # Get all unique RDF pairs from the reference data
    ref_pairs = list(created_rdfs["ref"].keys()) if created_rdfs["ref"] else []

    models_data = {}

    for model_name in MODELS:
        if rdf_scores.get(model_name) is None or created_rdfs.get(model_name) is None:
            continue

        model_data = {}
        ref_data = created_rdfs["ref"]
        model_rdf_data = created_rdfs[model_name]

        # Get scores for each RDF pair
        rdf_errors = aml.compute_all_errors(ref_data, model_rdf_data)

        model_data["metrics"] = {}
        model_data["metrics"]["rdf_score"] = {}
        pair_point_list = []
        for pair in ref_pairs:
            if pair in rdf_errors:
                error_data = rdf_errors[pair]
                d_score = aml.error_score_percentage(error_data[2])  # MAE score

                pair_point_list.append(
                    {
                        "label": pair,
                        "value": d_score,
                        "data": {
                            "x_values": error_data[0].tolist(),  # r values
                            "ref": ref_data[pair][1].tolist(),  # reference RDF
                            "pred": model_rdf_data[pair][1].tolist(),  # predicted RDF
                            "error": error_data[1].tolist(),
                            "ylabel": "g(r)",
                            "xlabel": "Distance (Å)",  # error array
                            "xlim": [0, 0.6],
                        },
                    }
                )
        model_data["metrics"]["rdf_score"] = pair_point_list
        models_data[model_name] = model_data

    # Create metrics mapping
    metrics = {}
    metrics["rdf_score"] = "RDF Score"

    return {"models": models_data, "metrics": metrics, "pairs": ref_pairs}


@pytest.fixture
@cell_to_bar(
    filename=OUT_PATH / "vdos_score_bar.json",
    x_label="Element",
    y_label="VDOS Score",
)
def build_vdos_interactive_data(
    vdos_scores: dict[str, list], created_vdos: dict[str, dict]
) -> dict:
    """
    Build interactive data structure for VDOS bar plot.

    Parameters
    ----------
    vdos_scores
        Average VDOS scores for all models.
    created_vdos
        Created VDOS for all models.

    Returns
    -------
    dict
        Interactive data structure for VDOS bar plot.
    """
    # Get all unique VDOS pairs from the reference data
    ref_pairs = list(created_vdos["ref"].keys()) if created_vdos["ref"] else []

    models_data = {}

    for model_name in MODELS:
        if vdos_scores.get(model_name) is None or created_vdos.get(model_name) is None:
            continue

        model_data = {}
        ref_data = created_vdos["ref"]
        model_vdos_data = created_vdos[model_name]

        # Get scores for each VDOS pair
        vdos_errors = aml.compute_all_errors(ref_data, model_vdos_data)

        model_data["metrics"] = {}
        model_data["metrics"]["vdos_score"] = {}
        pair_point_list = []
        for pair in ref_pairs:
            if pair in vdos_errors:
                error_data = vdos_errors[pair]
                d_score = aml.error_score_percentage(error_data[2])  # MAE score

                pair_point_list.append(
                    {
                        "label": pair,
                        "value": d_score,
                        "data": {
                            "x_values": error_data[0].tolist(),  # r values
                            "ref": ref_data[pair][1].tolist(),  # reference RDF
                            "pred": model_vdos_data[pair][1].tolist(),  # predicted RDF
                            "error": error_data[1].tolist(),
                            "ylabel": "Intensity (a.u)",
                            "xlabel": "Frequency (cm⁻¹)",  # error array
                            "xlim": [0, 4000],
                        },
                    }
                )
        model_data["metrics"]["vdos_score"] = pair_point_list
        models_data[model_name] = model_data

    # Create metrics mapping
    metrics = {}
    metrics["vdos_score"] = "VDOS Score"

    return {"models": models_data, "metrics": metrics, "pairs": ref_pairs}


# ----------------------+----------------------+---------------------- #
# ----------------------+-------- VACF --------+---------------------- #
# ----------------------+----------------------+---------------------- #


@pytest.fixture
def created_vacf() -> dict[str, dict]:
    """
    Create VACF for all models.

    Returns
    -------
    dict[str, dict]
        Dictionary of VACF for all models.
    """
    vacf = {"ref": None} | dict.fromkeys(MODELS)

    # Load reference RDF
    with open(f"{DATA_PATH}/vacf_reference.pkl", "rb") as f_in:
        vacf["ref"] = pickle.load(f_in)

    # Load topology
    ref_topology = f"{DATA_PATH}/init.pdb"

    # Check if vacf curves directory exists, if not create it
    if not VACF_CURVE_PATH.exists():
        VACF_CURVE_PATH.mkdir(parents=True, exist_ok=True)

    # Load model RDFs
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        # Check if vacf curves directory exists for model, if not create it
        if not (VACF_CURVE_PATH / model_name).exists():
            (VACF_CURVE_PATH / model_name).mkdir(parents=True, exist_ok=True)

        vel_xyz = model_dir / "md-velc.xyz"
        if not vel_xyz.exists():
            continue

        test_vel = mdt.load(vel_xyz, top=ref_topology, stride=1)
        # Truncate to same length as reference
        test_top = mdt.load_pdb(ref_topology)

        test_vel.unitcell_lengths = np.repeat(
            test_top.unitcell_lengths, len(test_vel), axis=0
        )
        test_vel.unitcell_vectors = np.repeat(
            test_top.unitcell_vectors, len(test_vel), axis=0
        )

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
                    ref_vel = mdt.load(REF_VEL_PATH, top=ref_topology, stride=1)
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
        with open(VACF_CURVE_PATH / model_name / "vacf_curves.pkl", "wb") as f_out:
            pickle.dump(vacf[model_name], f_out)

    return vacf


@pytest.fixture
def vacf_scores(created_vacf: dict[str, dict]) -> dict[str, float]:
    """
    Get Average VACF score for all models.

    Parameters
    ----------
    created_vacf
        Created VACF for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Average VACF scores for all models.
    """
    results = {"ref": []} | dict.fromkeys(MODELS)
    store_ref = False
    for model_name in MODELS:
        if created_vacf[model_name] is None:
            continue
        vacf_errors = aml.compute_all_errors_vacf(
            created_vacf[f"ref_{model_name}"], created_vacf[model_name]
        )
        # Mae is stored per pair in [2] slot of tuple
        vacf_scores = [
            aml.error_score_percentage(error[2]) for error in vacf_errors.values()
        ]
        results[model_name] = vacf_scores
        if not store_ref:
            results["ref"] = [100.0 for _ in vacf_scores]

        store_ref = True

    return results


@pytest.fixture
@cell_to_bar(
    filename=OUT_PATH / "vacf_score_bar.json",
    x_label="Element",
    y_label="VACF Score",
)
def build_vacf_interactive_data(
    vacf_scores: dict[str, list], created_vacf: dict[str, dict]
) -> dict:
    """
    Build interactive data structure for VACF bar plot.

    Parameters
    ----------
    vacf_scores
        Average VACF scores for all models.
    created_vacf
        Created VACF for all models.

    Returns
    -------
    dict
        Interactive data structure for VACF bar plot.
    """
    # Get all unique VACF pairs from the reference data
    ref_pairs = list(created_vacf["ref"].keys()) if created_vacf["ref"] else []

    models_data = {}

    for model_name in MODELS:
        if vacf_scores.get(model_name) is None or created_vacf.get(model_name) is None:
            continue

        model_data = {}
        ref_data = created_vacf[f"ref_{model_name}"]
        model_vacf_data = created_vacf[model_name]

        # Get scores for each VACF pair
        vacf_errors = aml.compute_all_errors_vacf(ref_data, created_vacf[model_name])

        model_data["metrics"] = {}
        model_data["metrics"]["vacf_score"] = {}
        pair_point_list = []
        for pair in ref_pairs:
            if pair in vacf_errors:
                error_data = vacf_errors[pair]
                d_score = aml.error_score_percentage(error_data[2])  # MAE score
                pair_point_list.append(
                    {
                        "label": pair,
                        "value": d_score,
                        "data": {
                            "x_values": error_data[0].tolist(),  # r values
                            "ref": ref_data[pair][1].tolist(),  # reference RDF
                            "pred": model_vacf_data[pair][1].tolist(),  # predicted RDF
                            "error": error_data[1].tolist(),
                            "ylabel": "VACF",
                            "xlabel": "Time [fs]",  # error array
                        },
                    }
                )
        model_data["metrics"]["vacf_score"] = pair_point_list
        models_data[model_name] = model_data

    # Create metrics mapping
    metrics = {}
    metrics["vacf_score"] = "VACF Score"

    return {"models": models_data, "metrics": metrics, "pairs": ref_pairs}


@pytest.fixture
def mean_vacf_score(vacf_scores: dict[str, float]) -> dict[str, float]:
    """
    Get Mean VACF score for all models.

    Parameters
    ----------
    vacf_scores
        Average VACF scores for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Mean VACF scores for all models.
    """
    mean_results = {"ref": 100.0} | dict.fromkeys(MODELS)
    for model_name, scores in vacf_scores.items():
        if scores is None:
            continue
        mean_results[model_name] = np.mean(scores)
    return mean_results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ice_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    mean_rdf_score: dict[str, float],
    mean_vdos_score: dict[str, float],
    mean_vacf_score: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    mean_rdf_score
        Mean RDF score for all models.
    mean_vdos_score
        Mean VDOS score for all models.
    mean_vacf_score
        Mean VACF score for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "rdf_score": mean_rdf_score,
        "vdos_score": mean_vdos_score,
        "vacf_score": mean_vacf_score,
    }


def test_ice(
    metrics: dict[str, dict],
    build_rdf_interactive_data: dict,
    build_vdos_interactive_data: dict,
    build_vacf_interactive_data: dict,
) -> None:
    """
    Run ice tests.

    Parameters
    ----------
    metrics
        All ice metrics.
    build_rdf_interactive_data
        Interactive data for RDF bar plot.
    build_vdos_interactive_data
        Interactive data for VDOS bar plot.
    build_vacf_interactive_data
        Interactive data for VACF bar plot.
    """
    return


"""@plot_parity(
    filename=OUT_PATH / "figure_rdf_score.json",
    title="RDF",
    x_label="Predicted RDF score",
    y_label="Reference RDF score",
    hoverdata={'RDF-pair': get_rdf_keys()},
)"""
