"""Analyse amorphous carbon melt-quench benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ase.io import read, write
from ase.neighborlist import neighbor_list
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table
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
STRUCTURES_DIR = OUT_PATH / "structures"


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


def _coordination_classes(atoms, cutoff: float = SP3_CUTOFF) -> np.ndarray:
    """
    Map coordination numbers to classes for coloring.

    Returns
    -------
    np.ndarray
        Array of integers: 0=sp1, 1=sp2, 2=sp3, -1=other.
    """
    i_indices, _ = neighbor_list("ij", atoms, cutoff)
    counts = np.bincount(i_indices, minlength=len(atoms))
    classes = np.full(len(atoms), -1, dtype=int)
    classes[counts == 2] = 0
    classes[counts == 3] = 1
    classes[counts == 4] = 2
    return classes


def _write_colored_structure(atoms, model_name: str, density: float) -> Path:
    """
    Write a colored structure file with coordination classes.

    Parameters
    ----------
    atoms
        Atomic configuration.
    model_name
        Model identifier.
    density
        Density value.

    Returns
    -------
    Path
        Path to the written structure file.
    """
    classes = _coordination_classes(atoms)
    atoms_copy = atoms.copy()
    atoms_copy.set_array("coordination", classes)
    # Map coordination to element symbols for color in viewer
    symbol_map = {
        0: "Cl",  # sp1 -> green
        1: "N",   # sp2 -> blue
        2: "P",   # sp3 -> orange
        -1: "C",
    }
    atoms_copy.symbols = [symbol_map[int(cls)] for cls in classes]
    out_dir = STRUCTURES_DIR / model_name / f"density_{density:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"colored_density_{density:.1f}.extxyz"
    write(out_path, atoms_copy, format="extxyz")
    return out_path


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
        _write_colored_structure(atoms, model_name, density)

    return results


def _structure_asset_path(model_name: str, density: float) -> str:
    """Return asset path for trajectory if present, else colored structure."""
    rel_dir = (
        Path("amorphous_materials")
        / "amorphous_carbon_melt_quench"
        / "structures"
        / model_name
        / f"density_{density:.1f}"
    )
    traj_name = f"trajectory_density_{density:.1f}.extxyz"
    colored_name = f"colored_density_{density:.1f}.extxyz"
    traj_path = STRUCTURES_DIR / rel_dir / traj_name
    if traj_path.exists():
        return f"assets/{rel_dir.as_posix()}/{traj_name}"
    return f"assets/{rel_dir.as_posix()}/{colored_name}"


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


def build_sp3_vs_density_plot(
    filename: Path,
    title: str = "sp3 fraction vs density",
    x_label: str = "Density (g cm^-3)",
    y_label: str = "sp3 count (%)",
) -> None:
    """
    Build sp3 vs density plot with experimental points as scatter only.

    Parameters
    ----------
    filename
        Path to save the plotly JSON file.
    title
        Plot title.
    x_label
        X-axis label.
    y_label
        Y-axis label.
    """
    data = sp3_vs_density()
    fig = go.Figure()

    dft = data.get("DFT")
    if dft:
        fig.add_trace(
            go.Scatter(
                x=dft[0],
                y=dft[1],
                name="DFT",
                mode="lines+markers",
            )
        )

    expt = data.get("Expt.")
    if expt:
        fig.add_trace(
            go.Scatter(
                x=expt[0],
                y=expt[1],
                name="Expt.",
                mode="markers",
                marker={"symbol": "cross", "size": 8},
            )
        )

    for model_name, series in data.items():
        if model_name in {"DFT", "Expt."}:
            continue
        customdata = [
            _structure_asset_path(model_name, density) for density in series[0]
        ]
        fig.add_trace(
            go.Scatter(
                x=series[0],
                y=series[1],
                name=model_name,
                mode="lines+markers",
                customdata=customdata,
            )
        )

    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": x_label}},
        yaxis={"title": {"text": y_label}},
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.write_json(filename)


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
    build_sp3_vs_density_plot(OUT_PATH / "figure_sp3_vs_density.json")
    return
