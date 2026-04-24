"""Analyse scaling_pol benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
import numpy as np
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "electric_field" / "scaling_pol" / "outputs"
OUT_PATH = APP_ROOT / "data" / "electric_field" / "scaling_pol"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


def get_system_names() -> list[str]:
    """
    Get list of scaling_pol system names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_file = model_dir / "ORCA_DATA.xyz"
            if xyz_file:
                mols = read(xyz_file, index=':')
                for mol in mols:
                    if np.linalg.norm(mol.info['external_field']) > 0:
                        system_names.append(mol.get_chemical_formula()+'_ef')
                    else:
                        system_names.append(mol.get_chemical_formula())
                break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_field_response.json",
    title="scaling_pol Field response",
    x_label="Predicted field response / kJ/mol",
    y_label="Reference field response / kJ/mol",
    hoverdata={
        "System": get_system_names(),
    },
)
def total_energies() -> dict[str, list]:
    """
    Get electric field response for all scaling_pol systems.



    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted electric field responses.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_file = model_dir / 'ORCA_DATA.xyz'
        if not xyz_file:
            continue

        # Read data
        mols = read(xyz_file, index=":")
    
        # Temporary dict storing total energies under electric field
        electric_field_total_energies = {
            "ref":{},
            model_name:{}
        }

        # Get total energies under electric field
        for mol in mols:
            # Get molecule energy
            molecule_energy = mol.get_potential_energy()
            name = str(mol.get_chemical_formula())

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)

            # Store energies and reference energies in a dict, and xyz files
            if any(mol.info['external_field']):
                electric_field_total_energies[model_name][name] = molecule_energy
                electric_field_total_energies["ref"][name] = mol.info["REF_energy"]
                write(structs_dir / f"{name}_field.xyz", mol)


        # Calculate electric field response
        for mol in mols:
            # Get molecule energy
            molecule_energy = mol.get_potential_energy()
            name = str(mol.get_chemical_formula())

            # Calculate field response, and store xyz files 
            if not any(mol.info['external_field']):
                diff = molecule_energy - electric_field_total_energies[model_name][name]
                ref_diff = mol.info["REF_energy"] - electric_field_total_energies["ref"][name]
                write(structs_dir / f"{name}.xyz", mol)

                # Save electric field response. Save reference only first time.
                results[model_name].append(diff * EV_TO_KJ_PER_MOL)
                if not ref_stored:
                    results["ref"].append(ref_diff * EV_TO_KJ_PER_MOL)

        ref_stored = True
    return results


@pytest.fixture
def scaling_pol_errors(total_energies) -> dict[str, float]:
    """
    Get mean absolute error for the electric field response.

    Parameters
    ----------
    total_energies
        Dictionary of reference and predicted electric field responses.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted electric field response errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if total_energies[model_name]:
            results[model_name] = mae(
                total_energies["ref"], total_energies[model_name]
            )
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "scaling_pol_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(scaling_pol_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all scaling_pol metrics.

    Parameters
    ----------
    scaling_pol_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": scaling_pol_errors,
    }


def test_scaling_pol(metrics: dict[str, dict]) -> None:
    """
    Run scaling_pol test.

    Parameters
    ----------
    metrics
        All scaling_pol metrics.
    """
    return