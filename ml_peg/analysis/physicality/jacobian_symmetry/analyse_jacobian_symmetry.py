"""Analyse Jacobian symmetry (lambda) benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "jacobian_symmetry" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "jacobian_symmetry"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EPS = 1e-3  # must match calc_jacobian_symmetry.EPS

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)


def build_jacobian(model_name: str, struct_name: str) -> np.ndarray | None:
    """
    Reconstruct the finite-difference Jacobian for one model/structure pair.

    Parameters
    ----------
    model_name
        Name of model to build the Jacobian for.
    struct_name
        Name of structure to build the Jacobian for.

    Returns
    -------
    np.ndarray | None
        The Jacobian matrix, or None if outputs are missing.
    """
    xyz_path = CALC_PATH / model_name / f"{struct_name}.xyz"
    if not xyz_path.exists():
        return None

    structs = read(xyz_path, index=":")
    n_dof = 3 * len(structs[0])

    forces_plus: dict[int, np.ndarray] = {}
    forces_minus: dict[int, np.ndarray] = {}
    for struct in structs:
        forces = struct.get_forces().flatten()
        if struct.info["sign"] == 1:
            forces_plus[struct.info["dof"]] = forces
        else:
            forces_minus[struct.info["dof"]] = forces

    jac = np.empty((n_dof, n_dof))
    for dof in range(n_dof):
        jac[:, dof] = (forces_plus[dof] - forces_minus[dof]) / (2 * EPS)

    return jac


def compute_lambda(jac: np.ndarray) -> float:
    """
    Get the fraction of the Jacobian's Frobenius norm that is antisymmetric.

    Parameters
    ----------
    jac
        Jacobian matrix.

    Returns
    -------
    float
        Lambda value, 0 for conservative forces, 1 for no conservative component.
    """
    jac_anti = (jac - jac.T) / 2
    return float(np.linalg.norm(jac_anti) / np.linalg.norm(jac))


def plot_lambda_by_structure(model_name: str, per_struct: dict[str, float]) -> None:
    """
    Build and save a bar chart of lambda per structure for one model.

    Parameters
    ----------
    model_name
        Name of model to plot.
    per_struct
        Lambda value for each structure.
    """
    structures = list(per_struct)
    values = [per_struct[name] for name in structures]

    fig = go.Figure(go.Bar(x=structures, y=values))
    fig.update_layout(
        title=f"Jacobian antisymmetry per structure: {model_name}",
        xaxis_title="Structure",
        yaxis_title="lambda",
    )

    # Mark structures the model could not evaluate (e.g. unsupported element),
    # so an empty bar is not mistaken for a perfect (zero) score.
    for name, value in zip(structures, values, strict=True):
        if value is None or np.isnan(value):
            fig.add_annotation(
                x=name,
                y=0,
                text="N/A",
                showarrow=False,
                yshift=12,
                font={"color": "#d62728", "size": 14},
            )

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.write_json(str(OUT_PATH / f"figure_{model_name}_lambda_by_structure.json"))


def plot_janti_heatmap(
    model_name: str, struct_name: str, jac: np.ndarray, symbols: list[str]
) -> None:
    """
    Build and save a heatmap of the antisymmetric Jacobian for one structure.

    Shows the antisymmetric part of the force Jacobian, (J - J.T) / 2, on a
    diverging colourscale centred on zero: a conservative model is near-blank,
    while non-conservative forces light up, revealing which degrees of freedom
    break the symmetry.

    Parameters
    ----------
    model_name
        Name of model.
    struct_name
        Name of structure.
    jac
        Jacobian matrix for the structure.
    symbols
        Chemical symbols of the atoms, used to label each degree of freedom.
    """
    jac_anti = (jac - jac.T) / 2
    labels = [
        f"{sym}{i}_{axis}" for i, sym in enumerate(symbols) for axis in ("x", "y", "z")
    ]

    # Symmetric colour range centred on 0; fall back to 1 for empty/NaN cases.
    vmax = float(np.nanmax(np.abs(jac_anti)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig = go.Figure(
        go.Heatmap(
            z=jac_anti,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-vmax,
            zmax=vmax,
        )
    )
    fig.update_layout(
        title=f"Antisymmetric Jacobian: {model_name} / {struct_name}",
        xaxis_title="Perturbed coordinate",
        yaxis_title="Force component",
        width=650,
        height=650,
        plot_bgcolor="#ffffff",
    )
    # Square cells: the Jacobian is square (n_dof x n_dof), and antisymmetry is
    # a mirror across the diagonal, which only reads cleanly with equal aspect.
    # dtick=1 forces a label on every degree of freedom (Plotly otherwise thins
    # them); the small tick font keeps all labels legible for larger structures.
    fig.update_xaxes(tickmode="linear", dtick=1, tickfont={"size": 8})
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        tickmode="linear",
        dtick=1,
        tickfont={"size": 8},
    )

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.write_json(str(OUT_PATH / f"figure_{model_name}_{struct_name}_janti.json"))


@pytest.fixture
def lambda_by_structure() -> dict[str, dict[str, float]]:
    """
    Get lambda for every (model, structure) pair.

    Returns
    -------
    dict[str, dict[str, float]]
        Lambda values for all structures, for all models.
    """
    results: dict[str, dict[str, float]] = {}
    for model_name in MODELS:
        results[model_name] = {}
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        for xyz_path in sorted(model_dir.glob("*.xyz")):
            struct_name = xyz_path.stem
            jac = build_jacobian(model_name, struct_name)
            if jac is not None:
                results[model_name][struct_name] = compute_lambda(jac)
                symbols = read(xyz_path, index=0).get_chemical_symbols()
                plot_janti_heatmap(model_name, struct_name, jac, symbols)

        if results[model_name]:
            plot_lambda_by_structure(model_name, results[model_name])

    return results


@pytest.fixture
def mean_lambda(
    lambda_by_structure: dict[str, dict[str, float]],
) -> dict[str, float | None]:
    """
    Get the mean lambda across all structures, for all models.

    Parameters
    ----------
    lambda_by_structure
        Lambda values for all structures, for all models.

    Returns
    -------
    dict[str, float | None]
        Mean lambda for all models.
    """
    results = {}
    for model_name, per_struct in lambda_by_structure.items():
        values = [v for v in per_struct.values() if not np.isnan(v)]
        results[model_name] = float(np.mean(values)) if values else None
    return results


@pytest.fixture
def max_lambda(
    lambda_by_structure: dict[str, dict[str, float]],
) -> dict[str, float | None]:
    """
    Get the max lambda across all structures, for all models.

    Parameters
    ----------
    lambda_by_structure
        Lambda values for all structures, for all models.

    Returns
    -------
    dict[str, float | None]
        Max lambda for all models.
    """
    results = {}
    for model_name, per_struct in lambda_by_structure.items():
        values = [v for v in per_struct.values() if not np.isnan(v)]
        results[model_name] = float(np.max(values)) if values else None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "jacobian_symmetry_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    mean_lambda: dict[str, float | None], max_lambda: dict[str, float | None]
) -> dict[str, dict]:
    """
    Get all Jacobian symmetry metrics.

    Parameters
    ----------
    mean_lambda
        Mean lambda across all structures, for all models.
    max_lambda
        Max lambda across all structures, for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "mean lambda": mean_lambda,
        "max lambda": max_lambda,
    }


def test_jacobian_symmetry(metrics: dict[str, dict]) -> None:
    """
    Run Jacobian symmetry analysis.

    Parameters
    ----------
    metrics
        All Jacobian symmetry metrics.
    """
    return
