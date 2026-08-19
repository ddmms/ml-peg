"""Analyse elemental slab oxygen adsorption benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_periodic_table
from ml_peg.analysis.utils.utils import (
    get_struct_info,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "surfaces" / "elemental_slab_oxygen_adsorption" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "elemental_slab_oxygen_adsorption"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Extract system metadata from mock calculation (filenames)
SYSTEM_INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    index="0",
    include_filenames=True,
    out_path=OUT_PATH,
)

# Each slab is elemental, so frame 0 of each file has a single element
ELEMENTS = [elements[0] for elements in SYSTEM_INFO["elements"]]

# Transition metals commonly given a Hubbard U correction in GGA+U training data
U_ELEMENTS = frozenset({"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"})


def compute_adsorption_energy(
    surface_e: float, mol_surf_e: float, molecule_e: float
) -> float:
    """
    Compute adsorption energy.

    Parameters
    ----------
    surface_e
        Energy of the clean surface.
    mol_surf_e
        Energy of the molecule+surface system.
    molecule_e
        Energy of the isolated molecule.

    Returns
    -------
    float
        Adsorption energy.
    """
    return mol_surf_e - (surface_e + molecule_e)


@pytest.fixture
def adsorption_energies() -> dict[str, list]:
    """
    Get adsorption energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted adsorption energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        for system_path in sorted((CALC_PATH / model_name).glob("*.xyz")):
            structs = read(system_path, index=":")

            surface, mol_surface, molecule = structs

            # Get predicted energies
            surface_e = surface.get_potential_energy()
            mol_surf_e = mol_surface.get_potential_energy()
            molecule_e = molecule.get_potential_energy()
            pred_ads_energy = compute_adsorption_energy(
                surface_e, mol_surf_e, molecule_e
            )
            results[model_name].append(pred_ads_energy)

            # Get reference energies (only store once)
            if not ref_stored:
                ref_surface_e = surface.info["ref_energy"]
                ref_mol_surf_e = mol_surface.info["ref_energy"]
                ref_molecule_e = molecule.info["ref_energy"]
                ref_ads_energy = compute_adsorption_energy(
                    ref_surface_e, ref_mol_surf_e, ref_molecule_e
                )
                results["ref"].append(ref_ads_energy)

            # Only write the first struct (slab+oxygen)
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system_path.stem}.xyz", structs[1])

        ref_stored = True
    return results


@pytest.fixture
def periodic_tables(adsorption_energies: dict[str, list]) -> None:
    """
    Write per-model periodic tables of adsorption energy errors.

    Parameters
    ----------
    adsorption_energies
        Dictionary of reference and predicted adsorption energies.
    """
    ref_energies = adsorption_energies["ref"]

    # Share one colour scale across models, wide enough that no error is clipped
    limit = max(
        (
            abs(pred - ref)
            for model_name in MODELS
            if adsorption_energies[model_name]
            for pred, ref in zip(
                adsorption_energies[model_name], ref_energies, strict=True
            )
        ),
        default=DEFAULT_THRESHOLDS["MAE"]["bad"],
    )

    ref_hover = {
        element: f"{energy:.3f}"
        for element, energy in zip(ELEMENTS, ref_energies, strict=True)
    }

    for model_name in MODELS:
        pred_energies = adsorption_energies[model_name]
        if not pred_energies:
            continue

        errors = {
            element: pred - ref
            for element, pred, ref in zip(
                ELEMENTS, pred_energies, ref_energies, strict=True
            )
        }
        plot_periodic_table(
            title=f"Adsorption energy error - {model_name}",
            colorbar_title="Error / eV",
            hoverdata={
                "Reference / eV": ref_hover,
                "Predicted / eV": {
                    element: f"{energy:.3f}"
                    for element, energy in zip(ELEMENTS, pred_energies, strict=True)
                },
            },
            filename=str(
                OUT_PATH / model_name / "adsorption_error_periodic_table.json"
            ),
            colorscale="RdBu",
            zmin=-limit,
            zmax=limit,
            # Errors within the "good" threshold stay near the midpoint colour,
            # so outliers remain distinguishable rather than saturating the scale
            symlog_scale=DEFAULT_THRESHOLDS["MAE"]["good"],
        )(lambda values=errors: values)()


def _subset_mae(
    adsorption_energies: dict[str, list], elements: set[str] | None = None
) -> dict[str, float]:
    """
    Get mean absolute error for adsorption energies of selected elements.

    Parameters
    ----------
    adsorption_energies
        Dictionary of reference and predicted adsorption energies.
    elements
        Elements to include. Default is `None`, corresponding to all elements.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted adsorption energy errors for all models.
    """
    mask = [elements is None or element in elements for element in ELEMENTS]
    refs = [
        energy
        for energy, keep in zip(adsorption_energies["ref"], mask, strict=True)
        if keep
    ]

    results = {}
    for model_name in MODELS:
        if not adsorption_energies[model_name]:
            results[model_name] = np.nan
            continue
        preds = [
            energy
            for energy, keep in zip(adsorption_energies[model_name], mask, strict=True)
            if keep
        ]
        results[model_name] = mae(refs, preds) if preds else np.nan
    return results


@pytest.fixture
def adsorption_mae(adsorption_energies: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for adsorption energies.

    Parameters
    ----------
    adsorption_energies
        Dictionary of reference and predicted adsorption energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted adsorption energy errors for all models.
    """
    return _subset_mae(adsorption_energies)


@pytest.fixture
def adsorption_mae_u(adsorption_energies: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for adsorption energies of Hubbard U elements.

    Parameters
    ----------
    adsorption_energies
        Dictionary of reference and predicted adsorption energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted adsorption energy errors for all models.
    """
    return _subset_mae(adsorption_energies, U_ELEMENTS)


@pytest.fixture
def adsorption_mae_non_u(adsorption_energies: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for adsorption energies of non-Hubbard U elements.

    Parameters
    ----------
    adsorption_energies
        Dictionary of reference and predicted adsorption energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted adsorption energy errors for all models.
    """
    return _subset_mae(adsorption_energies, set(ELEMENTS) - U_ELEMENTS)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "elemental_slab_oxygen_adsorption_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    adsorption_mae: dict[str, float],
    adsorption_mae_u: dict[str, float],
    adsorption_mae_non_u: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    adsorption_mae
        Mean absolute errors for all models.
    adsorption_mae_u
        Mean absolute errors for Hubbard U elements, for all models.
    adsorption_mae_non_u
        Mean absolute errors for non-Hubbard U elements, for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": adsorption_mae,
        "U MAE": adsorption_mae_u,
        "non-U MAE": adsorption_mae_non_u,
    }


def test_elemental_slab_oxygen_adsorption(
    metrics: dict[str, dict], periodic_tables: None
) -> None:
    """
    Run elemental_slab_oxygen_adsorption test.

    Parameters
    ----------
    metrics
        All elemental slab oxygen adsorption metrics.
    periodic_tables
        Per-model periodic-table heatmaps (side-effect only).
    """
    return
