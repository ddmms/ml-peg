"""Analysis of copper water interface benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils import aml_md_analysis as aml
from ml_peg.analysis.utils import md_water_analysis as md
from ml_peg.analysis.utils.decorators import build_table, cell_to_bar, plot_hist
from ml_peg.analysis.utils.dipoles import get_z_dipoles
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "surfaces" / "copper_water_interface" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "copper_water_interface"
RDF_CURVE_PATH = OUT_PATH / "rdf_curves"
VDOS_CURVE_PATH = OUT_PATH / "vdos_curves"
VACF_CURVE_PATH = OUT_PATH / "vacf_curves"


METRIC_LABELS = {
    "rdf_score": "RDF Score",
    "vdos_score": "VDOS Score",
    "vacf_score": "VACF Score",
    "stdev_dipole_z_deviation": "Dipole Moment Stdev Deviation",
}
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATA_PATH = (
    download_s3_data(
        filename="copper_water_interface.zip",
        key="inputs/surfaces/copper_water_interface/copper_water_interface.zip",
    )
    / "copper_water_interface"
)
REF_VEL_PATH = DATA_PATH / "pbe-d3-md-vel.xyz"
REF_DIPOLE_PATH = DATA_PATH / "ref_dipole_data.npy"

# Dipole histogram binning (matches the reference dipole distribution range).
DIPOLE_HIST_BINS = {"start": -0.05, "end": 0.05, "size": 0.0025}

# Oxygen point-charge magnitude (e) for the water dipole, shared with the
# water_slab_dipoles benchmark for a consistent dipole definition.
DIPOLE_CHARGE = 0.5562


# ----------------------+----------------------+---------------------- #
# ----------------------+-------- RDF ---------+---------------------- #
# ----------------------+----------------------+---------------------- #


@pytest.fixture
def created_rdfs() -> dict[str, dict]:
    """
    Create RDFs for all models.

    Returns
    -------
    dict[str, dict]
        Dictionary of RDFs for all models.
    """
    return md.create_rdfs(MODELS, DATA_PATH, CALC_PATH, RDF_CURVE_PATH)


@pytest.fixture
def rdf_scores(created_rdfs: dict[str, dict]) -> dict[str, float]:
    """
    Get Average RDF score for all models.

    Parameters
    ----------
    created_rdfs
        Created RDFs for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Average RDF scores for all models.
    """
    return md.property_scores(
        created_rdfs, MODELS, errors_fn=aml.compute_all_errors, ref_key=lambda m: "ref"
    )


@pytest.fixture
def mean_rdf_score(rdf_scores: dict[str, float]) -> dict[str, float]:
    """
    Get Mean RDF score for all models.

    Parameters
    ----------
    rdf_scores
        Average RDF scores for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Mean RDF scores for all models.
    """
    return md.mean_score(rdf_scores, MODELS)


@pytest.fixture
@cell_to_bar(
    filename=OUT_PATH / "rdf_score_bar.json",
    x_label="X Coord",
    y_label="RDF Score",
)
def build_rdf_interactive_data(
    rdf_scores: dict[str, list], created_rdfs: dict[str, dict]
) -> dict:
    """
    Build interactive data structure for RDF bar plot.

    Parameters
    ----------
    rdf_scores
        Average RDF scores for all models.
    created_rdfs
        Created RDFs for all models.

    Returns
    -------
    dict
        Interactive data structure for RDF bar plot.
    """
    return md.build_bar_data(
        rdf_scores,
        created_rdfs,
        MODELS,
        metric_key="rdf_score",
        metric_label="RDF Score",
        ylabel="g(r)",
        xlabel="Distance (Å)",
        errors_fn=aml.compute_all_errors,
        ref_key=lambda m: "ref",
        xlim=[0, 0.6],
    )


# ----------------------+----------------------+---------------------- #
# ----------------------+-------- VDOS --------+---------------------- #
# ----------------------+----------------------+---------------------- #


@pytest.fixture
def created_vdos() -> dict[str, dict]:
    """
    Create VDOS for all models.

    Returns
    -------
    dict[str, dict]
        Dictionary of VDOS for all models.
    """
    return md.create_vdos(MODELS, DATA_PATH, CALC_PATH, VDOS_CURVE_PATH)


@pytest.fixture
def vdos_scores(created_vdos: dict[str, dict]) -> dict[str, float]:
    """
    Get Average VDOS score for all models.

    Parameters
    ----------
    created_vdos
        Created VDOS for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Average VDOS scores for all models.
    """
    return md.property_scores(
        created_vdos, MODELS, errors_fn=aml.compute_all_errors, ref_key=lambda m: "ref"
    )


@pytest.fixture
def mean_vdos_score(vdos_scores: dict[str, float]) -> dict[str, float]:
    """
    Get Mean VDOS score for all models.

    Parameters
    ----------
    vdos_scores
        Average VDOS scores for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Mean VDOS scores for all models.
    """
    return md.mean_score(vdos_scores, MODELS)


@pytest.fixture
@cell_to_bar(
    filename=OUT_PATH / "vdos_score_bar.json",
    x_label="Element",
    y_label="VDOS Score",
)
def build_vdos_interactive_data(
    vdos_scores: dict[str, list], created_vdos: dict[str, dict]
) -> dict:
    """
    Build interactive data structure for VDOS bar plot.

    Parameters
    ----------
    vdos_scores
        Average VDOS scores for all models.
    created_vdos
        Created VDOS for all models.

    Returns
    -------
    dict
        Interactive data structure for VDOS bar plot.
    """
    return md.build_bar_data(
        vdos_scores,
        created_vdos,
        MODELS,
        metric_key="vdos_score",
        metric_label="VDOS Score",
        ylabel="Intensity (a.u)",
        xlabel="Frequency (cm⁻¹)",
        errors_fn=aml.compute_all_errors,
        ref_key=lambda m: "ref",
        xlim=[0, 4000],
    )


# ----------------------+----------------------+---------------------- #
# ----------------------+------- Dipole -------+---------------------- #
# ----------------------+----------------------+---------------------- #


@pytest.fixture
def stdev_dipole_z_deviation() -> dict[str, dict]:
    """
    Get standard deviation difference of dipole moments of water molecules.

    Returns
    -------
    dict[str, dict]
        Scored per-model dipole standard-deviation deviations plus the raw
        per-model dipole arrays used to render the distribution histogram.
    """
    dipole_results = {"ref": None} | dict.fromkeys(MODELS)
    raw_dipoles = {}

    # Load reference dipole data.
    # NOTE: ref_dipole_data.npy must be regenerated with the same point-charge q
    # (DIPOLE_CHARGE) used below for the deviation to be on a consistent scale.
    ref_dipoles = np.load(REF_DIPOLE_PATH)
    ref_dipole_stdev = np.std(ref_dipoles)
    dipole_results["ref"] = ref_dipole_stdev

    for model_name in MODELS:
        model_traj = CALC_PATH / model_name / "md-traj.extxyz"
        if not model_traj.exists():
            continue

        # Compute the per-frame total z-dipole per unit area from the trajectory,
        # the same way as the water_slab_dipoles benchmark.
        frames = read(model_traj, ":")
        dipoles_z = get_z_dipoles(frames, q=DIPOLE_CHARGE)

        dipole_results[model_name] = np.abs(np.std(dipoles_z) - ref_dipole_stdev)
        raw_dipoles[model_name] = dipoles_z.tolist()

    # Add reference dipoles for the overlaid reference histogram
    raw_dipoles["ref"] = ref_dipoles.tolist()

    return {"stdev_dipole_z_deviation": dipole_results, "raw_dipoles": raw_dipoles}


@pytest.fixture
@plot_hist(
    bins=DIPOLE_HIST_BINS,
    title="Dipole Moment Distribution",
    x_label="Pz/A [e/Å]",
    y_label="Density",
    filename=OUT_PATH / "figure_hist_dipoles.json",
)
def build_dipole_histogram(
    stdev_dipole_z_deviation: dict[str, dict],
) -> dict[str, list]:
    """
    Build dipole moment histogram data structure.

    Parameters
    ----------
    stdev_dipole_z_deviation
        Standard deviation difference of dipole moments of water molecules.

    Returns
    -------
    dict[str, list]
        Raw per-model dipole arrays for the distribution histogram.
    """
    return stdev_dipole_z_deviation["raw_dipoles"]


# ----------------------+----------------------+---------------------- #
# ----------------------+-------- VACF --------+---------------------- #
# ----------------------+----------------------+---------------------- #


@pytest.fixture
def created_vacf() -> dict[str, dict]:
    """
    Create VACF for all models.

    Returns
    -------
    dict[str, dict]
        Dictionary of VACF for all models.
    """
    return md.create_vacf(MODELS, DATA_PATH, CALC_PATH, VACF_CURVE_PATH, REF_VEL_PATH)


@pytest.fixture
def vacf_scores(created_vacf: dict[str, dict]) -> dict[str, float]:
    """
    Get Average VACF score for all models.

    Parameters
    ----------
    created_vacf
        Created VACF for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Average VACF scores for all models.
    """
    return md.property_scores(
        created_vacf,
        MODELS,
        errors_fn=aml.compute_all_errors_vacf,
        ref_key=lambda m: f"ref_{m}",
    )


@pytest.fixture
def mean_vacf_score(vacf_scores: dict[str, float]) -> dict[str, float]:
    """
    Get Mean VACF score for all models.

    Parameters
    ----------
    vacf_scores
        Average VACF scores for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Mean VACF scores for all models.
    """
    return md.mean_score(vacf_scores, MODELS)


@pytest.fixture
@cell_to_bar(
    filename=OUT_PATH / "vacf_score_bar.json",
    x_label="Element",
    y_label="VACF Score",
)
def build_vacf_interactive_data(
    vacf_scores: dict[str, list], created_vacf: dict[str, dict]
) -> dict:
    """
    Build interactive data structure for VACF bar plot.

    Parameters
    ----------
    vacf_scores
        Average VACF scores for all models.
    created_vacf
        Created VACF for all models.

    Returns
    -------
    dict
        Interactive data structure for VACF bar plot.
    """
    return md.build_bar_data(
        vacf_scores,
        created_vacf,
        MODELS,
        metric_key="vacf_score",
        metric_label="VACF Score",
        ylabel="VACF",
        xlabel="Time [fs]",
        errors_fn=aml.compute_all_errors_vacf,
        ref_key=lambda m: f"ref_{m}",
    )


# ----------------------+----------------------+---------------------- #
# ----------------------+------- Metrics ------+---------------------- #
# ----------------------+----------------------+---------------------- #


@pytest.fixture
@build_table(
    filename=OUT_PATH / "copper_water_interface_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    mean_rdf_score: dict[str, float],
    mean_vdos_score: dict[str, float],
    mean_vacf_score: dict[str, float],
    stdev_dipole_z_deviation: dict[str, dict],
    build_dipole_histogram: dict[str, list],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    mean_rdf_score
        Mean RDF score for all models.
    mean_vdos_score
        Mean VDOS score for all models.
    mean_vacf_score
        Mean VACF score for all models.
    stdev_dipole_z_deviation
        Standard deviation difference of dipole moments for all models.
    build_dipole_histogram
        Dipole moment histogram data for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "rdf_score": mean_rdf_score,
        "vdos_score": mean_vdos_score,
        "vacf_score": mean_vacf_score,
        "stdev_dipole_z_deviation": stdev_dipole_z_deviation[
            "stdev_dipole_z_deviation"
        ],
    }


def test_copper_water_interface(
    metrics: dict[str, dict],
    build_rdf_interactive_data: dict,
    build_vdos_interactive_data: dict,
    build_vacf_interactive_data: dict,
    build_dipole_histogram: dict[str, list],
) -> None:
    """
    Run copper_water_interface tests.

    Parameters
    ----------
    metrics
        All copper_water_interface metrics.
    build_rdf_interactive_data
        Interactive data for RDF bar plot.
    build_vdos_interactive_data
        Interactive data for VDOS bar plot.
    build_vacf_interactive_data
        Interactive data for VACF bar plot.
    build_dipole_histogram
        Dipole moment histogram data for all models.
    """
    return
