"""Analyse bond dissociation energy benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
import pytest
from scipy import stats

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "molecular" / "BDEs" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "BDEs"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KCAL_PER_MOL = units.mol / units.kcal

# Populated by dft_geometry_bdes / mlff_geometry_bdes; shared with downstream
# fixtures without appearing in the returned dict (which would add a scatter trace).
_COMPOUND_LABELS: list[str] = []
_MLFF_COMPOUND_LABELS: list[str] = []


def into_dict_of_labels(atoms, key):
    """
    Group atoms based on their atoms.info[key] labels.

    Parameters
    ----------
    atoms : list[Atoms]
        Atoms objects with info[key] labels to group.
    key : str
        Key in atoms.info to group by.

    Returns
    -------
    dict[str, list[Atoms]]
        Dictionary mapping each label value to a list of Atoms objects.
    """
    dict_out = {}
    for at in atoms:
        label = at.info[key]
        if label not in dict_out:
            dict_out[label] = []
        dict_out[label].append(at)
    return dict_out


def get_bde(mol_energy, rad_energy, isolated_h_energy):
    """
    Compute bond dissociation energy.

    Parameters
    ----------
    mol_energy : float
        Energy of the intact molecule.
    rad_energy : float
        Energy of the molecule with the hydrogen atom removed.
    isolated_h_energy : float
        Energy of the isolated hydrogen atom.

    Returns
    -------
    float
        Bond dissociation energy.
    """
    return rad_energy + isolated_h_energy - mol_energy


def get_hover_data_labels(key) -> list[str]:
    """
    Get labels based on the key.

    Parameters
    ----------
    key : str
        Key in atoms.info to retrieve labels from.

    Returns
    -------
    list[str]
        List of system names from atoms.info entries.
    """
    labels = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                for xyz_file in xyz_files:
                    atoms = read(xyz_file, ":")
                    labels += [at.info[key] for at in atoms]
                break
    return labels


def group_values_by_labels(
    values: list[float], labels: list[str]
) -> dict[str, list[float]]:
    """
    Group values by their corresponding string labels.

    Parameters
    ----------
    values : list[float]
        Values to group.
    labels : list[str]
        Labels corresponding to each value.

    Returns
    -------
    dict[str, list[float]]
        Dictionary mapping each label to a list of its values.
    """
    results = {}
    for label, value in zip(labels, values, strict=False):
        if label not in results:
            results[label] = []
        results[label].append(value)

    return results


def process_bdes_per_compound(all_atoms, prefix):
    """
    Compute BDEs for given atoms and property prefix.

    Parameters
    ----------
    all_atoms
        List of Atoms with appropriate labels for radicals,
        molecules and isolated atoms.
    prefix
        "dft_" or "pred_" to fetch the appropriate energy.

    Returns
    -------
    dict[str: list[float]]
        List of computed bond dissociation energies, for each compound.
    """
    compound_dict = into_dict_of_labels(all_atoms, key="compound")

    isolated_atoms = compound_dict.pop("isolated_atom")
    isolated_atoms_dict = into_dict_of_labels(isolated_atoms, "element")
    isolated_h_energy = isolated_atoms_dict["H"][0].info[f"{prefix}energy"]

    all_bdes = {}
    for compound_name, compound_atoms in compound_dict.items():
        mol_rad_dict = into_dict_of_labels(compound_atoms, "mol_or_rad")
        if "rad" not in mol_rad_dict:
            continue

        all_bdes[compound_name] = []

        mol = mol_rad_dict["mol"]
        assert len(mol) == 1
        mol = mol[0]
        mol_energy = mol.info[f"{prefix}energy"]

        for rad in mol_rad_dict["rad"]:
            rad_energy = rad.info[f"{prefix}energy"]
            bde = get_bde(
                mol_energy=mol_energy,
                rad_energy=rad_energy,
                isolated_h_energy=isolated_h_energy,
            )
            all_bdes[compound_name].append(bde * EV_TO_KCAL_PER_MOL)

    return all_bdes


def mean_kendalls_tau(ref_by_label, pred_by_label):
    """
    Compute mean Kendall's tau across labels.

    Parameters
    ----------
    ref_by_label : dict[str, list]
        Reference ranks grouped by compound label.
    pred_by_label : dict[str, list]
        Predicted ranks grouped by compound label.

    Returns
    -------
    float
        Mean Kendall's tau correlation across all labels.
    """
    kendalls_taus = []
    for label in ref_by_label.keys():
        ref = ref_by_label[label]
        pred = pred_by_label[label]
        correlation = stats.kendalltau(ref, pred).correlation
        kendalls_taus.append(correlation)
    return np.mean(kendalls_taus)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_bdes(xyz_suffix: str, labels_out: list) -> dict[str, list]:
    """
    Load BDEs from output xyz files for all models.

    Parameters
    ----------
    xyz_suffix
        Suffix of the xyz filename, e.g. "dft_opt" or "mlff_opt".
    labels_out
        List to populate in-place with compound labels for each BDE entry.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted BDEs keyed by model name and "ref".
    """
    xyz_filename = f"cytochrome_p450_substrates.{xyz_suffix}.xyz"
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    labels_out.clear()
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob(xyz_filename))
        if not xyz_files:
            continue

        for xyz_file in xyz_files:
            all_atoms = read(xyz_file, index=":")
            pred_bdes_by_label = process_bdes_per_compound(all_atoms, "pred_")
            for compound_label, pred_bdes in pred_bdes_by_label.items():
                if not ref_stored:
                    labels_out += [compound_label] * len(pred_bdes)
                results[model_name] += pred_bdes

            if not ref_stored:
                dft_bdes_by_label = process_bdes_per_compound(all_atoms, "dft_")
                for dft_bdes in dft_bdes_by_label.values():
                    results["ref"] += dft_bdes
        ref_stored = True

    return results


def _compute_ranks(bdes: dict, labels: list) -> dict[str, list]:
    """
    Compute BDE ranks per compound for all models and reference.

    Parameters
    ----------
    bdes
        Dictionary of reference and predicted BDEs keyed by model name and "ref".
    labels
        Compound label for each BDE entry.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted BDE ranks.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    for model_name in results.keys():
        bdes_by_compound = group_values_by_labels(
            values=bdes[model_name],
            labels=labels,
        )
        for compound_bdes in bdes_by_compound.values():
            ranks = np.argsort(np.argsort(compound_bdes))
            results[model_name] += list(ranks)
    return results


def _compute_bde_errors(bdes: dict) -> dict[str, float]:
    """
    Compute mean absolute error of predicted BDEs against reference.

    Parameters
    ----------
    bdes
        Dictionary of reference and predicted BDEs keyed by model name and "ref".

    Returns
    -------
    dict[str, float]
        MAE for each model.
    """
    return {
        model_name: mae(bdes["ref"], bdes[model_name]) if bdes[model_name] else None
        for model_name in MODELS
    }


def _save_struct_files(xyz_suffix: str) -> None:
    """
    Save individual radical XYZ files for app visualisation.

    Writes one file per BDE data point (one per radical) in the same order
    as ``_load_bdes``, so that scatter-point indices map to structure files.
    Saves for every model that has the relevant output file.

    Parameters
    ----------
    xyz_suffix
        Suffix of the xyz filename, e.g. ``"dft_opt"`` or ``"mlff_opt"``.
    """
    xyz_filename = f"cytochrome_p450_substrates.{xyz_suffix}.xyz"
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue
        xyz_files = sorted(model_dir.glob(xyz_filename))
        if not xyz_files:
            continue

        struct_dir = OUT_PATH / model_name / xyz_suffix
        struct_dir.mkdir(parents=True, exist_ok=True)

        for xyz_file in xyz_files:
            all_atoms = read(xyz_file, index=":")
            compound_dict = into_dict_of_labels(all_atoms, key="compound")
            compound_dict.pop("isolated_atom", None)

            idx = 0
            for compound_atoms in compound_dict.values():
                mol_rad_dict = into_dict_of_labels(compound_atoms, "mol_or_rad")
                if "rad" not in mol_rad_dict:
                    continue
                for rad in mol_rad_dict["rad"]:
                    write(struct_dir / f"{idx}.xyz", rad)
                    idx += 1


def _compute_rank_correlations(ranks: dict, labels: list) -> dict[str, float]:
    """
    Compute mean Kendall's tau rank correlation against reference ranks.

    Parameters
    ----------
    ranks
        Dictionary of reference and predicted BDE ranks keyed by model name and "ref".
    labels
        Compound label for each rank entry.

    Returns
    -------
    dict[str, float]
        Mean Kendall's tau for each model.
    """
    ref_ranks_by_compound = group_values_by_labels(
        values=ranks["ref"],
        labels=labels,
    )
    results = {}
    for model_name in MODELS:
        if ranks[model_name]:
            pred_ranks_by_compound = group_values_by_labels(
                values=ranks[model_name],
                labels=labels,
            )
            results[model_name] = mean_kendalls_tau(
                ref_by_label=ref_ranks_by_compound,
                pred_by_label=pred_ranks_by_compound,
            )
        else:
            results[model_name] = None
    return results


# ---------------------------------------------------------------------------
# DFT-geometry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure.CYP3A4.dft_opt_geometry.BDEs.json",
    title="Bond Dissociation Energies on DFT Geometries",
    x_label="Predicted BDE / kcal/mol",
    y_label="Reference BDE / kcal/mol",
    hoverdata={
        "Compound": get_hover_data_labels("compound"),
    },
)
def dft_geometry_bdes() -> dict[str, list]:
    """
    Compute BDEs for all sp3 H atoms on DFT optimised geometries.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted BDEs and labels of the
        compound for the corresponding BDE.
    """
    result = _load_bdes("dft_opt", _COMPOUND_LABELS)
    _save_struct_files("dft_opt")
    return result


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure.CYP3A4.dft_opt_geometry.BDE_ranks.json",
    title="Bond Dissociation Energy ranks on DFT Geometries",
    x_label="Predicted BDE rank",
    y_label="Reference BDE rank",
    hoverdata={
        "Compound": get_hover_data_labels("compound"),
    },
)
def dft_geometry_bde_ranks(dft_geometry_bdes) -> dict[str, list]:
    """
    Compute BDE ranks for all sp3 H atoms using DFT optimised geometries.

    Parameters
    ----------
    dft_geometry_bdes : dict[str, list]
        Dictionary of reference and predicted BDEs.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted BDE ranks.
    """
    return _compute_ranks(dft_geometry_bdes, _COMPOUND_LABELS)


@pytest.fixture
def dft_geometry_bde_errors(dft_geometry_bdes) -> dict[str, float]:
    """
    Get mean absolute error for bond dissociation energies.

    Parameters
    ----------
    dft_geometry_bdes
        Dictionary of reference and predicted BDEs.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted BDE errors for all models.
    """
    return _compute_bde_errors(dft_geometry_bdes)


@pytest.fixture
def dft_geometry_bde_rank_correlations(dft_geometry_bde_ranks) -> dict[str, float]:
    """
    Compute mean Kendall's tau rank correlation across all molecules.

    For each molecule, sp3 C-H bond strengths are ranked from lowest to
    highest by both DFT and MLIP, and the correlation is measured by
    Kendall's tau. The final metric is the average across all molecules.

    Parameters
    ----------
    dft_geometry_bde_ranks : dict[str, list]
        Dictionary of reference and predicted BDE ranks.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted BDE rank correlation for all models.
    """
    return _compute_rank_correlations(dft_geometry_bde_ranks, _COMPOUND_LABELS)


# ---------------------------------------------------------------------------
# MLFF-geometry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure.CYP3A4.mlff_opt_geometry.BDEs.json",
    title="Bond Dissociation Energies on MLFF Geometries",
    x_label="Predicted BDE / kcal/mol",
    y_label="Reference BDE / kcal/mol",
    hoverdata={
        "Compound": get_hover_data_labels("compound"),
    },
)
def mlff_geometry_bdes() -> dict[str, list]:
    """
    Compute BDEs for all sp3 H atoms on MLFF optimised geometries.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted BDEs and labels of the
        compound for the corresponding BDE.
    """
    result = _load_bdes("mlff_opt", _MLFF_COMPOUND_LABELS)
    _save_struct_files("mlff_opt")
    return result


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure.CYP3A4.mlff_opt_geometry.BDE_ranks.json",
    title="Bond Dissociation Energy ranks on MLFF Geometries",
    x_label="Predicted BDE rank",
    y_label="Reference BDE rank",
    hoverdata={
        "Compound": get_hover_data_labels("compound"),
    },
)
def mlff_geometry_bde_ranks(mlff_geometry_bdes) -> dict[str, list]:
    """
    Compute BDE ranks for all sp3 H atoms using MLFF optimised geometries.

    Parameters
    ----------
    mlff_geometry_bdes : dict[str, list]
        Dictionary of reference and predicted BDEs.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted BDE ranks.
    """
    return _compute_ranks(mlff_geometry_bdes, _MLFF_COMPOUND_LABELS)


@pytest.fixture
def mlff_geometry_bde_errors(mlff_geometry_bdes) -> dict[str, float]:
    """
    Get mean absolute error for bond dissociation energies on MLFF geometries.

    Parameters
    ----------
    mlff_geometry_bdes
        Dictionary of reference and predicted BDEs.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted BDE errors for all models.
    """
    return _compute_bde_errors(mlff_geometry_bdes)


@pytest.fixture
def mlff_geometry_bde_rank_correlations(mlff_geometry_bde_ranks) -> dict[str, float]:
    """
    Compute mean Kendall's tau rank correlation across all molecules on MLFF geometries.

    For each molecule, sp3 C-H bond strengths are ranked from lowest to
    highest by both DFT and MLIP, and the correlation is measured by
    Kendall's tau. The final metric is the average across all molecules.

    Parameters
    ----------
    mlff_geometry_bde_ranks : dict[str, list]
        Dictionary of reference and predicted BDE ranks.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted BDE rank correlation for all models.
    """
    return _compute_rank_correlations(mlff_geometry_bde_ranks, _MLFF_COMPOUND_LABELS)


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------


@pytest.fixture
@build_table(
    filename=OUT_PATH / "metrics_table.CYP3A4.dft_opt_geometry.BDEs.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    dft_geometry_bde_errors: dict[str, float],
    dft_geometry_bde_rank_correlations: dict[str, float],
    mlff_geometry_bde_errors: dict[str, float],
    mlff_geometry_bde_rank_correlations: dict[str, float],
) -> dict[str, dict]:
    """
    Get all BDE metrics.

    Parameters
    ----------
    dft_geometry_bde_errors
        Mean absolute errors on DFT-relaxed structures.
    dft_geometry_bde_rank_correlations
        Mean Kendall's tau across predicted and reference BDE ranks on DFT geometries.
    mlff_geometry_bde_errors
        Mean absolute errors on MLFF-relaxed structures.
    mlff_geometry_bde_rank_correlations
        Mean Kendall's tau across predicted and reference BDE ranks on MLFF geometries.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Direct BDE": dft_geometry_bde_errors,
        "BDE rank": dft_geometry_bde_rank_correlations,
        "Direct BDE (MLFF opt)": mlff_geometry_bde_errors,
        "BDE rank (MLFF opt)": mlff_geometry_bde_rank_correlations,
    }


def test_bdes(metrics: dict[str, dict]) -> None:
    """
    Run BDEs test.

    Parameters
    ----------
    metrics
        All BDEs metrics.
    """
    return
