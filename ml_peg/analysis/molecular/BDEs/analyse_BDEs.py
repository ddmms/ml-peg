"""Analyse X23 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read
import pytest

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
    Compute BDEs for all sp3 H atoms using DFT optimised geometries.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted BDEs.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob("*xyz"))
        if not xyz_files:
            continue

        for xyz_file in xyz_files:
            all_atoms = read(xyz_file, index=":")
            pred_bdes = process_bdes(all_atoms, "pred_")
            results[model_name] += pred_bdes

            if not ref_stored:
                dft_bdes = process_bdes(all_atoms, "dft_")
                results["ref"] += dft_bdes
        ref_stored = True

    return results


def process_bdes(all_atoms, prefix):
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
    list[float]
        List of computed bond dissociation energies.
    """
    compound_dict = into_dict_of_labels(all_atoms, key="compound")

    isolated_atoms = compound_dict.pop("isolated_atom")
    isolated_atoms_dict = into_dict_of_labels(isolated_atoms, "element")
    isolated_h_energy = isolated_atoms_dict["H"][0].info[f"{prefix}energy"]

    all_bdes = []
    for compound_atoms in compound_dict.values():
        mol_rad_dict = into_dict_of_labels(compound_atoms, "mol_or_rad")
        if "rad" not in mol_rad_dict:
            continue

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
            all_bdes.append(bde * EV_TO_KCAL_PER_MOL)

    return all_bdes


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
    results = {}
    for model_name in MODELS:
        if dft_geometry_bdes[model_name]:
            results[model_name] = mae(
                dft_geometry_bdes["ref"],
                dft_geometry_bdes[model_name],
            )
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "metrics_table.CYP3A4.dft_opt_geometry.BDEs.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(dft_geometry_bde_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all total rad energy metrics.

    Parameters
    ----------
    dft_geometry_bde_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": dft_geometry_bde_errors,
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
