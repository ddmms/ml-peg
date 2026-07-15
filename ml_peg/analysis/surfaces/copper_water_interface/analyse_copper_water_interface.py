"""Analysis of copper water interface benchmark."""

from __future__ import annotations

from pathlib import Path
import pickle

from ase.io import read
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.surfaces.copper_water_interface.density_profile import (
    create_density_profiles,
)
from ml_peg.analysis.utils import aml_md_analysis as aml
from ml_peg.analysis.utils import md_water_analysis as md
from ml_peg.analysis.utils.decorators import build_table, cell_to_bar, plot_scatter
from ml_peg.analysis.utils.dipoles import (
    get_z_dipoles,
    get_z_dipoles_average_integrated_profile,
    get_z_dipoles_frames,
)
from ml_peg.analysis.utils.utils import load_metrics_config, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "surfaces" / "copper_water_interface" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "copper_water_interface"
RDF_CURVE_PATH = OUT_PATH / "rdf_curves"
VDOS_CURVE_PATH = OUT_PATH / "vdos_curves"
VACF_CURVE_PATH = OUT_PATH / "vacf_curves"
DENSITY_CURVE_PATH = OUT_PATH / "density_curves"


METRIC_LABELS = {
    "rdf_score": "RDF Score",
    "vdos_score": "VDOS Score",
    "vacf_score": "VACF Score",
    "density_score": "Density Score",
    "dipole_profile_score": "Integrated Dipole Profile Score",
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
REF_DIPOLE_PATH = DATA_PATH / "ref_dipole_data.npy"

# Dipole histogram binning (e/Å). Tune the range and bar width here: "start"/"end"
# set the x-axis span and "size" the bar width.
DIPOLE_HIST_BINS = {"start": -0.1, "end": 0.1, "size": 0.001}

# Oxygen point-charge magnitude (e) for the water dipole, shared with the
# water_slab_dipoles benchmark for a consistent dipole definition.
DIPOLE_CHARGE = 0.5562

# z-dipole window (e/Å) outside which a frame is a "breakdown candidate" (its
# dipole is large enough to close the interfacial water band gap). Asymmetric,
# unlike the symmetric water_slab_dipoles threshold, and specific to the
# q=DIPOLE_CHARGE dipole scale.
DIPOLE_LOWER_BOUND = -0.019686563354143947
DIPOLE_UPPER_BOUND = 0.011680694256792076


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


def _write_dipole_histogram(raw_dipoles: dict[str, list]) -> None:
    """
    Write the dipole distribution histogram, coloured by the band-gap window.

    Each model bar is coloured green when its bin lies inside the stable window
    ``[DIPOLE_LOWER_BOUND, DIPOLE_UPPER_BOUND]`` and red when outside, so the plot
    matches the ``Fraction Breakdown Candidates`` metric. The reference is drawn as a
    neutral step outline for comparison rather than coloured bars.

    Parameters
    ----------
    raw_dipoles
        Raw per-model (and ``"ref"``) dipole arrays for the distribution histogram.
    """
    start = DIPOLE_HIST_BINS["start"]
    end = DIPOLE_HIST_BINS["end"]
    size = DIPOLE_HIST_BINS["size"]
    n_bins = int(round((end - start) / size))
    edges = start + size * np.arange(n_bins + 1)
    centers = edges[:-1] + size / 2

    good_color = "#2ca02c"  # green: dipole inside the stable band-gap window
    bad_color = "#d62728"  # red: dipole outside the window (breakdown candidate)
    bar_colors = [
        good_color if DIPOLE_LOWER_BOUND <= center <= DIPOLE_UPPER_BOUND else bad_color
        for center in centers
    ]

    fig = go.Figure()
    for model_name, dipoles in raw_dipoles.items():
        density, _ = np.histogram(np.asarray(dipoles), bins=edges, density=True)
        if model_name == "ref":
            # Reference: step outline so the coloured model bars stay readable.
            fig.add_trace(
                go.Scatter(
                    x=centers,
                    y=density,
                    mode="lines",
                    line_shape="hvh",
                    line={"color": "#333", "width": 1.5},
                    name="ref",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=density,
                    width=size,
                    marker_color=bar_colors,
                    marker_line_width=0,
                    name=model_name,
                    opacity=0.75,
                )
            )

    # Mark the window boundaries where the colour transitions.
    for bound in (DIPOLE_LOWER_BOUND, DIPOLE_UPPER_BOUND):
        fig.add_vline(x=bound, line={"color": "#888", "width": 1, "dash": "dash"})

    fig.update_layout(
        title={"text": "Dipole Moment Distribution"},
        xaxis={"title": {"text": "Total dipole / Unit Area (e/Å)"}},
        yaxis={"title": {"text": "Density"}},
        barmode="overlay",
        bargap=0,
    )

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.write_json(OUT_PATH / "figure_hist_dipoles.json")


@pytest.fixture
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
    raw_dipoles = stdev_dipole_z_deviation["raw_dipoles"]
    _write_dipole_histogram(raw_dipoles)
    return raw_dipoles


@pytest.fixture
def fraction_breakdown_candidates(
    stdev_dipole_z_deviation: dict[str, dict],
) -> dict[str, float]:
    """
    Get fraction of frames with z-dipole outside the stable band-gap window.

    Parameters
    ----------
    stdev_dipole_z_deviation
        Standard deviation difference of dipole moments of water molecules,
        carrying the raw per-model dipole arrays.

    Returns
    -------
    dict[str, float]
        Fraction of breakdown-candidate frames for the reference and each model.
    """
    raw = stdev_dipole_z_deviation["raw_dipoles"]
    results = {"ref": None} | dict.fromkeys(MODELS)
    for key in ("ref", *MODELS):
        if key not in raw:
            continue
        dipoles = np.asarray(raw[key])
        results[key] = (
            (dipoles < DIPOLE_LOWER_BOUND) | (dipoles > DIPOLE_UPPER_BOUND)
        ).sum() / len(dipoles)
    return results


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
    return md.create_vacf(MODELS, DATA_PATH, CALC_PATH, VACF_CURVE_PATH)


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
# ----------------------+------ Density -------+---------------------- #
# ----------------------+----------------------+---------------------- #


@pytest.fixture
def created_density() -> dict[str, dict]:
    """
    Create O and H density profiles for all models.

    Returns
    -------
    dict[str, dict]
        Dictionary of density profiles for all models.
    """
    return create_density_profiles(MODELS, DATA_PATH, CALC_PATH, DENSITY_CURVE_PATH)


@pytest.fixture
def density_scores(created_density: dict[str, dict]) -> dict[str, float]:
    """
    Get per-element density profile scores for all models.

    Parameters
    ----------
    created_density
        Created density profiles for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of per-element density scores for all models.
    """
    return md.property_scores(
        created_density,
        MODELS,
        errors_fn=aml.compute_all_errors,
        ref_key=lambda m: "ref",
    )


@pytest.fixture
def mean_density_score(density_scores: dict[str, float]) -> dict[str, float]:
    """
    Get Mean density profile score for all models.

    Parameters
    ----------
    density_scores
        Per-element density scores for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Mean density scores for all models.
    """
    return md.mean_score(density_scores, MODELS)


@pytest.fixture
@cell_to_bar(
    filename=OUT_PATH / "density_score_bar.json",
    x_label="Element",
    y_label="Density Score",
)
def build_density_interactive_data(
    density_scores: dict[str, list], created_density: dict[str, dict]
) -> dict:
    """
    Build interactive data structure for density profile bar plot.

    Parameters
    ----------
    density_scores
        Per-element density scores for all models.
    created_density
        Created density profiles for all models.

    Returns
    -------
    dict
        Interactive data structure for density profile bar plot.
    """
    return md.build_bar_data(
        density_scores,
        created_density,
        MODELS,
        metric_key="density_score",
        metric_label="Density Score",
        ylabel="Density",
        xlabel="z from surface (Å)",
        errors_fn=aml.compute_all_errors,
        ref_key=lambda m: "ref",
        xlim=[0, 35],
    )


# ----------------------+----------------------+---------------------- #
# ----------------------+--- Dipole Profile ---+---------------------- #
# ----------------------+----------------------+---------------------- #


@pytest.fixture
def created_dipole_profile() -> dict[str, dict]:
    """
    Create integrated dipole profiles for the reference and all models.

    Reuses the shared dipole primitives in :mod:`ml_peg.analysis.utils.dipoles`: the
    reference curve is loaded from the precomputed ``dipole_profile_reference.pkl`` and
    each model curve is computed from its ``md-traj.extxyz`` trajectory.

    Returns
    -------
    dict[str, dict]
        Dictionary of ``{"dipole": (z, profile)}`` for the reference and all models.
    """
    profiles = {"ref": None} | dict.fromkeys(MODELS)

    with open(DATA_PATH / "dipole_profile_reference.pkl", "rb") as f_in:
        profiles["ref"] = pickle.load(f_in)

    for model_name in MODELS:
        model_traj = CALC_PATH / model_name / "md-traj.extxyz"
        if not model_traj.exists():
            continue

        frames = read(model_traj, ":")
        _, all_dipoles = get_z_dipoles_frames(frames)
        z_bins, profile = get_z_dipoles_average_integrated_profile(
            all_dipoles, n_frames=len(frames)
        )
        profiles[model_name] = {"dipole": (z_bins, profile)}

    return profiles


@pytest.fixture
def dipole_profile_scores(created_dipole_profile: dict[str, dict]) -> dict[str, float]:
    """
    Get integrated dipole profile score for all models.

    Parameters
    ----------
    created_dipole_profile
        Created integrated dipole profiles for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of integrated dipole profile scores for all models.
    """
    return md.property_scores(
        created_dipole_profile,
        MODELS,
        errors_fn=aml.compute_all_errors,
        ref_key=lambda m: "ref",
    )


@pytest.fixture
def mean_dipole_profile_score(
    dipole_profile_scores: dict[str, float],
) -> dict[str, float]:
    """
    Get Mean integrated dipole profile score for all models.

    Parameters
    ----------
    dipole_profile_scores
        Integrated dipole profile scores for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Mean integrated dipole profile scores for all models.
    """
    return md.mean_score(dipole_profile_scores, MODELS)


@pytest.fixture
@plot_scatter(
    filename=OUT_PATH / "figure_dipole_profile.json",
    title="Integrated Dipole Profile",
    x_label="z from surface (Å)",
    y_label="Integrated dipole / Unit Area (e/Å)",
    show_line=True,
    show_markers=False,
)
def build_dipole_profile_plot(
    created_dipole_profile: dict[str, dict],
) -> dict[str, tuple]:
    """
    Build the integrated dipole profile line plot (reference + one line per model).

    Parameters
    ----------
    created_dipole_profile
        Created integrated dipole profiles for all models.

    Returns
    -------
    dict[str, tuple]
        Mapping of ``"ref"``/model name to its ``(z, profile)`` curve, as expected by
        the ``plot_scatter`` decorator.
    """
    return {"ref": created_dipole_profile["ref"]["dipole"]} | {
        model_name: created_dipole_profile[model_name]["dipole"]
        for model_name in MODELS
        if created_dipole_profile.get(model_name)
    }


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
    mean_density_score: dict[str, float],
    mean_dipole_profile_score: dict[str, float],
    stdev_dipole_z_deviation: dict[str, dict],
    fraction_breakdown_candidates: dict[str, float],
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
    mean_density_score
        Mean density profile score for all models.
    mean_dipole_profile_score
        Mean integrated dipole profile score for all models.
    stdev_dipole_z_deviation
        Standard deviation difference of dipole moments for all models.
    fraction_breakdown_candidates
        Fraction of breakdown-candidate frames for all models.
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
        "density_score": mean_density_score,
        "dipole_profile_score": mean_dipole_profile_score,
        "stdev_dipole_z_deviation": stdev_dipole_z_deviation[
            "stdev_dipole_z_deviation"
        ],
        "Fraction Breakdown Candidates": fraction_breakdown_candidates,
    }


def test_copper_water_interface(
    metrics: dict[str, dict],
    build_rdf_interactive_data: dict,
    build_vdos_interactive_data: dict,
    build_vacf_interactive_data: dict,
    build_density_interactive_data: dict,
    build_dipole_profile_plot: dict,
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
    build_density_interactive_data
        Interactive data for density profile bar plot.
    build_dipole_profile_plot
        Integrated dipole profile line plot data.
    build_dipole_histogram
        Dipole moment histogram data for all models.
    """
    # Save elemental info for element filtering (see developer guide: filter).
    write_struct_info(
        data_path=CALC_PATH / "mock" / "md-final.extxyz",
        out_path=OUT_PATH,
        index=0,
    )
