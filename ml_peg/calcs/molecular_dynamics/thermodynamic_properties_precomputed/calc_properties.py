"""
Stage precomputed thermodynamic properties of selected organic liquids.

Liquids include 146 chemicals from https://doi.org/10.1021/ct200731v
"""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
from typing import Any

import pytest

from ml_peg.calcs.molecular_dynamics.thermodynamic_properties.utils import (
    get_available_cas,
)
from ml_peg.calcs.utils.utils import BENCHMARK_DATA_DIR, download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


def get_precomputed_logs(model_name: str) -> Path:
    """
    Get the path to precomputed thermodynamic property logs.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the precomputed log files.
    """
    extracted_path = BENCHMARK_DATA_DIR / f"thermodynamic_properties_{model_name}"

    if extracted_path.exists():
        return extracted_path

    return (
        download_s3_data(
            filename=f"thermodynamic_properties_{model_name}.zip",
            key=(
                "inputs/molecular_dynamics/"
                "thermodynamic_properties_precomputed/"
                f"thermodynamic_properties_{model_name}.zip"
            ),
        )
        / f"thermodynamic_properties_{model_name}"
    )


def get_config_path(model_name: str) -> Path:
    """
    Get the path to thermodynamic property configurations.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the model configurations.
    """
    extracted_path = BENCHMARK_DATA_DIR / "thermodynamic_properties"

    if extracted_path.exists():
        return extracted_path

    return (
        download_s3_data(
            filename="thermodynamic_properties.zip",
            key=(
                "inputs/molecular_dynamics/"
                "thermodynamic_properties_precomputed/"
                "thermodynamic_properties.zip"
            ),
        )
        / f"thermodynamic_properties_{model_name}"
    )


def stage_precomputed_results(
    model_name: str,
    cas: str,
) -> bool:
    """
    Stage precomputed thermodynamic property results for one system.

    Parameters
    ----------
    model_name
        Name of the model.
    cas
        CAS number identifying the system.

    Returns
    -------
    bool
        Whether any precomputed files were staged.
    """
    config_path = get_config_path(model_name)
    logs_path = get_precomputed_logs(model_name)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(exist_ok=True, parents=True)

    staged = False

    for phase in ("liq", "gas"):
        label = f"{cas}-{phase}"

        log_file = logs_path / f"{label}.log"
        xyz_file = config_path / "equilibrated_structures_xyz" / f"{label}.xyz"

        if not log_file.exists():
            continue

        if not xyz_file.exists():
            raise FileNotFoundError(
                f"No XYZ configuration found for {label}: {xyz_file}"
            )

        copy2(log_file, out_dir / log_file.name)
        copy2(xyz_file, out_dir / xyz_file.name)

        staged = True

    return staged


@pytest.mark.framework("mace-off-24")
@pytest.mark.parametrize("mlip", MODELS.items())
def test_thermodynamic_properties_precomputed(
    mlip: tuple[str, Any],
    cas: str,
) -> None:
    """
    Stage precomputed thermodynamic property results.

    Parameters
    ----------
    mlip
        Name of model and model definition.
    cas
        CAS number identifier of the system.
    """
    model_name, _ = mlip

    if cas == "all":
        for selected_cas in get_available_cas():
            stage_precomputed_results(
                model_name=model_name,
                cas=selected_cas,
            )
        return

    stage_precomputed_results(
        model_name=model_name,
        cas=cas,
    )
