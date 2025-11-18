"""Analyse lattice constants benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "lattice_constants" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "lattice_constants"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_crystal_formulae() -> list[str]:
    """
    Get list of crystal formulae.

    Returns
    -------
    list[str]
        List of crystal formulae from structure files.
    """
    formulae = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue
        struct_files = sorted(model_dir.glob("*-traj.extxyz"))
        for struct_file in struct_files:
            atoms = read(struct_file)
            name = atoms.info["name"]
            if name == "SiC":
                formulae.extend(("SiC(a)", "SiC(c)"))
            else:
                formulae.append(name)
        break

    return formulae


FORMULAE = get_crystal_formulae()


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_consts_exp.json",
    title="Lattice constants",
    x_label="Predicted lattice constant / Å",
    y_label="Experimental lattice constant / Å",
    hoverdata={
        "Formula": FORMULAE,
    },
)
def lattice_constants_exp() -> dict[str, list]:
    """
    Get experimental and predicted lattice constant for all crystals.

    Returns
    -------
    dict[str, list]
        Dictionary of experimental and predicted lattice energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        struct_files = sorted(model_dir.glob("*-traj.extxyz"))
        if not struct_files:
            continue

        for struct_file in struct_files:
            structs = read(struct_file, index=":")

            formula = structs[-1].info["name"]
            lattice_type = structs[-1].info["lattice_type"]

            a_exp = structs[-1].info["a_exp"]
            a_pred = structs[-1].cell.lengths()[0]
            if formula == "SiC":
                c_exp = structs[-1].info["c_exp"]
                c_pred = structs[-1].cell.lengths()[2]
            else:
                c_exp = None
                c_pred = None

                if lattice_type in ("fcc", "diamond", "rocksalt", "zincblende"):
                    a_pred = a_pred * np.sqrt(2)
                elif lattice_type == "bcc":
                    a_pred = a_pred * 2 / np.sqrt(3)

            results[model_name].append(a_pred)
            if c_pred:
                results[model_name].append(c_pred)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(a_exp)
                if c_exp:
                    results["ref"].append(c_exp)

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{structs[-1].info['name']}.xyz", structs)

        ref_stored = True

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_consts_dft.json",
    title="Lattice constants",
    x_label="Predicted lattice constant / Å",
    y_label="DFT lattice constant / Å",
    hoverdata={
        "Formula": FORMULAE,
    },
)
def lattice_constants_dft() -> dict[str, list]:
    """
    Get DFT and predicted lattice constant for all crystals.

    Returns
    -------
    dict[str, list]
        Dictionary of DFT and predicted lattice constants.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        struct_files = sorted(model_dir.glob("*-traj.extxyz"))
        if not struct_files:
            continue

        for struct_file in struct_files:
            structs = read(struct_file, index=":")

            formula = structs[-1].info["name"]
            lattice_type = structs[-1].info["lattice_type"]

            a_dft = structs[-1].info["a_dft"]
            a_pred = structs[-1].cell.lengths()[0]
            if formula == "SiC":
                c_dft = structs[-1].info["c_dft"]
                c_pred = structs[-1].cell.lengths()[2]
            else:
                c_dft = None
                c_pred = None

                if lattice_type in ("fcc", "diamond", "rocksalt", "zincblende"):
                    a_pred = a_pred * np.sqrt(2)
                elif lattice_type == "bcc":
                    a_pred = a_pred * 2 / np.sqrt(3)

            results[model_name].append(a_pred)
            if c_pred:
                results[model_name].append(c_pred)

            # Store reference lattice constants (only once)
            if not ref_stored:
                results["ref"].append(a_dft)
                if c_dft:
                    results["ref"].append(c_dft)

        ref_stored = True

    return results


@pytest.fixture
def lattice_constant_exp_errors(lattice_constants_exp) -> dict[str, float]:
    """
    Get mean absolute error for lattice constants compared to experimental reference.

    Parameters
    ----------
    lattice_constants_exp
        Dictionary of experimental and predicted lattice constants.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted lattice constant errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            lattice_constants_exp["ref"], lattice_constants_exp[model_name]
        )
    return results


@pytest.fixture
def lattice_constant_dft_errors(lattice_constants_dft) -> dict[str, float]:
    """
    Get mean absolute error for lattice constants compared to DFT reference.

    Parameters
    ----------
    lattice_constants_dft
        Dictionary of DFT and predicted lattice constants.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted lattice constant errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            lattice_constants_dft["ref"], lattice_constants_dft[model_name]
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "lattice_constants_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    lattice_constant_exp_errors: dict[str, float],
    lattice_constant_dft_errors: dict[str, float],
) -> dict[str, dict]:
    """
    Get all lattice constant metrics.

    Parameters
    ----------
    lattice_constant_exp_errors
        Mean absolute errors.
    lattice_constant_dft_errors
        Mean absolute errors.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE (Experimental)": lattice_constant_exp_errors,
        "MAE (PBE)": lattice_constant_dft_errors,
    }


def test_lattice_constants(metrics: dict[str, dict]) -> None:
    """
    Run lattice constant test.

    Parameters
    ----------
    metrics
        All lattice constant metrics.
    """
    return
