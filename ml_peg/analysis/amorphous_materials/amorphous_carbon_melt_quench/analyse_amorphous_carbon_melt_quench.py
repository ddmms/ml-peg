"""Analyse amorphous carbon melt-quench benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ase.io import read
from ase.neighborlist import neighbor_list
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "amorphous_materials" / "amorphous_carbon_melt_quench"
OUT_PATH = APP_ROOT / "data" / "amorphous_materials" / "amorphous_carbon_melt_quench"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

REF_DFT_PATH = Path(__file__).with_name("reference_dft.csv")
REF_EXPT_PATH = Path(__file__).with_name("reference_expt.csv")

DENSITY_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]
SP3_CUTOFF = 1.85


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load reference data from a CSV file.

    Parameters
    ----------
    path
        Path to CSV file with columns: density, sp3_percent.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Densities and sp3 percentages.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Reference data not found: {path}. Provide a CSV with 'density' and "
            "'sp3_percent' columns."
        )

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    densities = data[:, 0]
    sp3 = data[:, 1]
    order = np.argsort(densities)
    return densities[order], sp3[order]


def _sp3_fraction_from_atoms(atoms, cutoff: float = SP3_CUTOFF) -> float:
    """Calculate sp3 fraction (%) from a structure."""
    i_indices, _ = neighbor_list("ij", atoms, cutoff)
    counts = np.bincount(i_indices, minlength=len(atoms))
    sp3_count = int(np.sum(counts == 4))
    return 100.0 * sp3_count / len(atoms)


def _load_model_summary(model_name: str) -> dict[float, float]:
    """
    Load sp3 fractions from final structures for a model.

    Parameters
    ----------
    model_name
        Model identifier.

    Returns
    -------
    dict[float, float]
        Mapping density -> sp3 fraction (%).
    """
    results: dict[float, float] = {}
    model_dir = CALC_PATH / "outputs" / model_name
    if not model_dir.exists():
        return results

    for density in DENSITY_GRID:
        final_path = (
            model_dir
            / f"density_{density:.1f}"
            / f"final_density_{density:.1f}.xyz"
        )
        if not final_path.exists():
            continue
        atoms = read(final_path)
        results[density] = _sp3_fraction_from_atoms(atoms)

    return results


def _load_all_model_data(models: Iterable[str]) -> dict[str, dict[float, float]]:
    """
    Load sp3 fraction data for all models.

    Parameters
    ----------
    models
        Iterable of model names.

    Returns
    -------
    dict[str, dict[float, float]]
        Mapping model -> {density: sp3_fraction}.
    """
    return {model: _load_model_summary(model) for model in models}


def _series_from_mapping(mapping: dict[float, float]) -> tuple[list[float], list[float]]:
    """Return density/sp3 lists aligned to the density grid."""
    densities: list[float] = []
    sp3: list[float] = []
    for density in DENSITY_GRID:
        if density in mapping:
            densities.append(density)
            sp3.append(mapping[density])
    return densities, sp3


def _mae_against_reference(
    model_series: dict[float, float], ref_density: np.ndarray, ref_sp3: np.ndarray
) -> float | None:
    """Compute MAE against a reference curve at the density grid."""
    densities, predictions = _series_from_mapping(model_series)
    if len(densities) != len(DENSITY_GRID):
        return None
    ref_interp = np.interp(densities, ref_density, ref_sp3)
    return float(mae(ref_interp.tolist(), predictions))


@plot_scatter(
    filename=OUT_PATH / "figure_sp3_vs_density.json",
    title="sp3 fraction vs density",
    x_label="Density (g cm^-3)",
    y_label="sp3 count (%)",
    show_line=True,
)
def sp3_vs_density() -> dict[str, tuple[list[float], list[float]]]:
    """Generate sp3 vs density plot data."""
    dft_density, dft_sp3 = _load_reference(REF_DFT_PATH)
    expt_density, expt_sp3 = _load_reference(REF_EXPT_PATH)

    results: dict[str, tuple[list[float], list[float]]] = {
        "DFT": (dft_density.tolist(), dft_sp3.tolist()),
        "Expt.": (expt_density.tolist(), expt_sp3.tolist()),
    }

    model_data = _load_all_model_data(MODELS)
    for model_name, mapping in model_data.items():
        densities, sp3 = _series_from_mapping(mapping)
        if densities:
            results[model_name] = (densities, sp3)

    return results


@pytest.fixture
def mae_vs_dft() -> dict[str, float | None]:
    """Compute MAE against DFT reference for each model."""
    dft_density, dft_sp3 = _load_reference(REF_DFT_PATH)
    model_data = _load_all_model_data(MODELS)
    results: dict[str, float | None] = {}
    for model_name, mapping in model_data.items():
        results[model_name] = _mae_against_reference(mapping, dft_density, dft_sp3)
    return results


@pytest.fixture
def mae_vs_expt() -> dict[str, float | None]:
    """Compute MAE against experimental reference for each model."""
    expt_density, expt_sp3 = _load_reference(REF_EXPT_PATH)
    model_data = _load_all_model_data(MODELS)
    results: dict[str, float | None] = {}
    for model_name, mapping in model_data.items():
        results[model_name] = _mae_against_reference(mapping, expt_density, expt_sp3)
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "amorphous_carbon_melt_quench_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    mae_vs_dft: dict[str, float | None],
    mae_vs_expt: dict[str, float | None],
) -> dict[str, dict]:
    """Build metrics table entries."""
    return {
        "MAE vs DFT": mae_vs_dft,
        "MAE vs Expt": mae_vs_expt,
    }


def test_amorphous_carbon_melt_quench(metrics: dict[str, dict]) -> None:
    """Run analysis for amorphous carbon melt-quench benchmark."""
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    sp3_vs_density()
    return
