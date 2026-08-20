"""
Analysis script for the aluminosilicates_densities benchmark.

Compares melt-quench glass densities from MD simulations against
experimental values from:
    Taylor, M. & Brown, G.E. (1979), Structure of mineral glasses -- I.
    The feldspar glasses NaAlSi3O8, KAlSi3O8, CaAl2Si2O8.
    Geochim. Cosmochim. Acta, doi:10.1016/0016-7037(79)90047-4

Experimental densities at room temperature (g/cm3):
    Albite    (NaAlSi3O8):  2.382
    Anorthite (CaAl2Si2O8): 2.691
    Sanidine  (KAlSi3O8):   2.366

Note: MD densities are expected to be systematically lower (~5-10%) than
experimental values due to the much faster quench rate in MD simulations
(5 K/ps vs K/min-K/h experimentally). This is a known and documented
artefact of melt-quench MD, common across all MLIP and classical MD studies.

Metrics are reported separately for each composition (albite, anorthite,
sanidine) to highlight composition-dependent model behaviour, e.g. models
that fail specifically for anorthite (Ca-rich, high Al/Si ratio) vs alkali
feldspars (Na/K, low Al/Si ratio).

MAE and MAPE are computed as the mean of per-replica errors (not the error
on the mean density), which is the statistically correct estimator.

Density extraction: from SLURM log files (slurm_*.out) -- the 'Final density' line
contains the mean density averaged over the 20 ps production run.
Only completed runs contribute a value.
FALLBACK:  from replica{N}_quenched.xyz atoms.info['density_final_gcm3'],
which also stores the mean over the production run (same value,
but requires the full run to have completed and the file written).
"""

from __future__ import annotations

import os
from pathlib import Path
import re

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CALC_PATH = Path(
    os.environ.get(
        "MLPEG_OUT_PATH",
        str(
            CALCS_ROOT / "molecular_dynamics" / "aluminosilicates_densities" / "outputs"
        ),
    )
)
LOG_PATH = CALC_PATH / "logs"

OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "aluminosilicates_densities"

MODELS = get_model_names(current_models)

COMPOSITIONS = ["albite", "anorthite", "sanidine"]
N_REPLICAS = 3

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# ---------------------------------------------------------------------------
# Experimental reference densities (g/cm3, room temperature)
# Source: Taylor & Brown (1979), GCA doi:10.1016/0016-7037(79)90047-4
# ---------------------------------------------------------------------------
EXP_DENSITY = {
    "albite": 2.382,
    "anorthite": 2.691,
    "sanidine": 2.366,
}

# Element filtering info for the app
INFO = get_struct_info(
    calc_path=CALC_PATH,
    model_name=MODELS[0] if MODELS else "mace-mp-0a",
    glob_pattern="*/*.xyz",
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
)


def get_densities_from_logs(model_name: str, composition: str) -> list[float]:
    """
    Read per-replica densities from log files.

    Each 'Final density' line corresponds to one completed replica.
    Values are means over the 20 ps NPT production run at 300 K.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model.
    composition : str
        Mineral composition ('albite', 'anorthite', 'sanidine').

    Returns
    -------
    list[float]
        Per-replica mean densities (g/cm3).
    """
    pattern = re.compile(
        rf"Final density \({re.escape(model_name)},\s*{re.escape(composition)}\):\s*"
        rf"([\d.]+)\s*\+/-"
    )
    densities = []
    if not LOG_PATH.exists():
        return densities
    for log_file in LOG_PATH.glob("slurm_*.out"):
        for line in log_file.read_text(errors="replace").splitlines():
            m = pattern.search(line)
            if m:
                densities.append(float(m.group(1)))
    return densities


def get_density_from_xyz(
    model_name: str, composition: str, replica: int
) -> float | None:
    """
    Read mean density from quenched structure file (fallback).

    Parameters
    ----------
    model_name : str
        Name of the MLIP model.
    composition : str
        Mineral composition.
    replica : int
        Replica index (0, 1, 2).

    Returns
    -------
    float or None
        Mean density in g/cm3, or None if file does not exist.
    """
    struct_path = (
        CALC_PATH / model_name / composition / f"replica{replica}_quenched.xyz"
    )
    if not struct_path.exists():
        return None
    atoms = read(str(struct_path))
    return atoms.info.get("density_final_gcm3", None)


def get_replica_densities(model_name: str, composition: str) -> list[float]:
    """
    Get per-replica densities for a model/composition pair.

    Tries logs first, falls back to xyz files.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model.
    composition : str
        Mineral composition.

    Returns
    -------
    list[float]
        Per-replica densities (g/cm3). Empty if no data available.
    """
    values = get_densities_from_logs(model_name, composition)
    if not values:
        values = [
            get_density_from_xyz(model_name, composition, r) for r in range(N_REPLICAS)
        ]
        values = [v for v in values if v is not None]
    return values


def compute_mae(model_name: str, composition: str) -> float:
    """
    MAE between simulated and experimental density for one model/composition.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model.
    composition : str
        Mineral composition.

    Returns
    -------
    float
        MAE in g/cm3, or NaN if no data available.
    """
    values = get_replica_densities(model_name, composition)
    if not values:
        return float("nan")
    ref = EXP_DENSITY[composition]
    return float(np.mean([abs(v - ref) for v in values]))


def compute_mape(model_name: str, composition: str) -> float:
    """
    MAPE between simulated and experimental density for one model/composition.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model.
    composition : str
        Mineral composition.

    Returns
    -------
    float
        MAPE in %, or NaN if no data available.
    """
    values = get_replica_densities(model_name, composition)
    if not values:
        return float("nan")
    ref = EXP_DENSITY[composition]
    return float(np.mean([abs(v - ref) / ref * 100 for v in values]))


def _mean_over_compositions(
    per_comp: dict[str, dict[str, float]],
    model_name: str,
) -> float:
    """
    Mean of a metric over all compositions, ignoring NaN.

    Parameters
    ----------
    per_comp : dict[str, dict[str, float]]
        Mapping of composition names to model metric dictionaries.
    model_name : str
        Name or identifier of the target model.

    Returns
    -------
    float
        Mean metric value across valid numeric entries, or NaN if none exist.
    """
    vals = [per_comp[comp][model_name] for comp in COMPOSITIONS]
    finite = [v for v in vals if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def copy_all_structures() -> None:
    """
    Wrap and save .xyz structures to the app output folder.

    Returns
    -------
    None
        This function does not return a value.
    """
    for model_name in MODELS:
        for comp in COMPOSITIONS:
            for replica in range(N_REPLICAS):
                src = CALC_PATH / model_name / comp / f"replica{replica}_quenched.xyz"
                if src.exists():
                    atoms = read(str(src))
                    atoms.wrap()
                    filename = f"{model_name}_{comp}_replica{replica}_quenched.xyz"
                    dst = OUT_PATH / model_name / comp / filename
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    write(str(dst), atoms)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_density_parity.json",
    title="Glass density: MD melt-quench vs experimental",
    x_label="Simulated density / g cm\u207b\u00b3",
    y_label="Experimental density / g cm\u207b\u00b3",
    hoverdata={"Composition": COMPOSITIONS},
)
def densities() -> dict[str, list]:
    """
    Collect simulated and experimental densities for the parity plot.

    Returns one value per composition per model (mean over replicas),
    used for display only. Metrics are computed from per-replica values.

    Returns
    -------
    dict[str, list]
        Reference and mean predicted densities.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    ref = [EXP_DENSITY[comp] for comp in COMPOSITIONS]
    results = {"ref": ref}
    for model_name in MODELS:
        model_densities = []
        for comp in COMPOSITIONS:
            values = get_replica_densities(model_name, comp)
            rho = float(np.mean(values)) if values else float("nan")
            model_densities.append(rho)
        results[model_name] = model_densities
    return results


@pytest.fixture
def density_mae_albite() -> dict[str, float]:
    """
    MAE of simulated vs experimental density for albite (g/cm3).

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to density MAE values.
    """
    return {m: compute_mae(m, "albite") for m in MODELS}


@pytest.fixture
def density_mae_anorthite() -> dict[str, float]:
    """
    MAE of simulated vs experimental density for anorthite (g/cm3).

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to density MAE values.
    """
    return {m: compute_mae(m, "anorthite") for m in MODELS}


@pytest.fixture
def density_mae_sanidine() -> dict[str, float]:
    """
    MAE of simulated vs experimental density for sanidine (g/cm3).

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to density MAE values.
    """
    return {m: compute_mae(m, "sanidine") for m in MODELS}


@pytest.fixture
def density_mae_mean(
    density_mae_albite: dict[str, float],
    density_mae_anorthite: dict[str, float],
    density_mae_sanidine: dict[str, float],
) -> dict[str, float]:
    """
    Mean MAE across all three compositions (g/cm3).

    Parameters
    ----------
    density_mae_albite : dict[str, float]
        Density MAE values for albite per model.
    density_mae_anorthite : dict[str, float]
        Density MAE values for anorthite per model.
    density_mae_sanidine : dict[str, float]
        Density MAE values for sanidine per model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to mean MAE values across all compositions.
    """
    per_comp = {
        "albite": density_mae_albite,
        "anorthite": density_mae_anorthite,
        "sanidine": density_mae_sanidine,
    }
    return {m: _mean_over_compositions(per_comp, m) for m in MODELS}


@pytest.fixture
def density_mape_albite() -> dict[str, float]:
    """
    MAPE of simulated vs experimental density for albite (%).

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to density MAPE values.
    """
    return {m: compute_mape(m, "albite") for m in MODELS}


@pytest.fixture
def density_mape_anorthite() -> dict[str, float]:
    """
    MAPE of simulated vs experimental density for anorthite (%).

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to density MAPE values.
    """
    return {m: compute_mape(m, "anorthite") for m in MODELS}


@pytest.fixture
def density_mape_sanidine() -> dict[str, float]:
    """
    MAPE of simulated vs experimental density for sanidine (%).

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to density MAPE values.
    """
    return {m: compute_mape(m, "sanidine") for m in MODELS}


@pytest.fixture
def density_mape_mean(
    density_mape_albite: dict[str, float],
    density_mape_anorthite: dict[str, float],
    density_mape_sanidine: dict[str, float],
) -> dict[str, float]:
    """
    Mean MAPE across all three compositions (%).

    Parameters
    ----------
    density_mape_albite : dict[str, float]
        Density MAPE values for albite per model.
    density_mape_anorthite : dict[str, float]
        Density MAPE values for anorthite per model.
    density_mape_sanidine : dict[str, float]
        Density MAPE values for sanidine per model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to mean MAPE values across all compositions.
    """
    per_comp = {
        "albite": density_mape_albite,
        "anorthite": density_mape_anorthite,
        "sanidine": density_mape_sanidine,
    }
    return {m: _mean_over_compositions(per_comp, m) for m in MODELS}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "aluminosilicates_densities_metrics_table.json",
    thresholds=DEFAULT_THRESHOLDS,
    metric_tooltips=DEFAULT_TOOLTIPS,
    weights=DEFAULT_WEIGHTS,
)
@pytest.fixture
@build_table(
    filename=OUT_PATH / "aluminosilicates_densities_metrics_table.json",
    thresholds=DEFAULT_THRESHOLDS,
    metric_tooltips=DEFAULT_TOOLTIPS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    density_mae_albite: dict[str, float],
    density_mae_anorthite: dict[str, float],
    density_mae_sanidine: dict[str, float],
    density_mae_mean: dict[str, float],
    density_mape_albite: dict[str, float],
    density_mape_anorthite: dict[str, float],
    density_mape_sanidine: dict[str, float],
    density_mape_mean: dict[str, float],
) -> dict[str, dict]:
    """
    Collect all benchmark metrics for the summary table.

    Parameters
    ----------
    density_mae_albite : dict[str, float]
        Density MAE values for albite.
    density_mae_anorthite : dict[str, float]
        Density MAE values for anorthite.
    density_mae_sanidine : dict[str, float]
        Density MAE values for sanidine.
    density_mae_mean : dict[str, float]
        Mean density MAE values across all compositions.
    density_mape_albite : dict[str, float]
        Density MAPE values for albite.
    density_mape_anorthite : dict[str, float]
        Density MAPE values for anorthite.
    density_mape_sanidine : dict[str, float]
        Density MAPE values for sanidine.
    density_mape_mean : dict[str, float]
        Mean density MAPE values across all compositions.

    Returns
    -------
    dict[str, dict]
        Dictionary containing all benchmark metrics mapped by metric name.
    """
    copy_all_structures()
    return {
        "Density MAE (albite)": density_mae_albite,
        "Density MAE (anorthite)": density_mae_anorthite,
        "Density MAE (sanidine)": density_mae_sanidine,
        "Density MAE (mean)": density_mae_mean,
        "Density MAPE (albite)": density_mape_albite,
        "Density MAPE (anorthite)": density_mape_anorthite,
        "Density MAPE (sanidine)": density_mape_sanidine,
        "Density MAPE (mean)": density_mape_mean,
    }


def test_aluminosilicates_densities_analysis(
    metrics: dict[str, dict],
    densities: dict[str, list],
) -> None:
    """
    Run aluminosilicates density benchmark analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Dictionary containing all benchmark summary metrics per model.
    densities : dict[str, list]
        Dictionary containing simulated and experimental density distributions.

    Returns
    -------
    None
        This test function does not return a value.
    """
    return
