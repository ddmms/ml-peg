"""Analyse HF neutron structure factor benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase.io import iread
import h5py
from MDANSE.Framework.Converters.Converter import Converter
from MDANSE.Framework.Jobs.IJob import IJob
from MDANSE.MolecularDynamics.Trajectory import Trajectory
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_MODEL_NAMES = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "superacids" / "HF_structure" / "outputs"
OUT_PATH = APP_ROOT / "data" / "superacids" / "HF_structure"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Time between trajectory frames, from the MD timestep and dump frequency
FRAME_SPACING_FS = 50.0

# Second half of the trajectory is production, sampled every STRIDE frames
PRODUCTION_FRACTION = 0.5
STRIDE = 2

# Minimum number of production frames for a structure factor to be computed
MIN_FRAMES = 10

# Real space grid for the pair distribution function, in nm.
# The cutoff is kept fixed for every model so that the Fourier transform is
# truncated identically, which would not be the case if it were derived from
# each model's own (NPT) cell.
R_MAX_NM = 0.6
R_STEP_NM = 0.005

# Reciprocal space grid, in 1/nm. Matches the experimental grid once converted
# to 1/A, so calculated and experimental points coincide.
Q_MIN_INV_NM = 7.5
Q_MAX_INV_NM = 100.0
Q_STEP_INV_NM = 0.5

INV_NM_TO_INV_ANG = 0.1
NM_TO_ANG = 10.0

# Upper bound of the range over which models are scored against experiment, in
# 1/A. The lower bound is the first experimental point. The first peak of S(q)
# is located within the same range.
SCORE_Q_MAX = 4.0


def load_reference_sq() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the experimental neutron structure factor.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Scattering vector in 1/A, and structure factor.
    """
    hf_structure_dir = (
        download_s3_data(
            key="inputs/superacids/HF_structure/HF_structure.zip",
            filename="HF_structure.zip",
        )
        / "HF_structure"
    )

    data = np.loadtxt(hf_structure_dir / "SQ_EXP.dat")

    return data[:, 0], data[:, 1]


def compute_sq(traj_path: Path, work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the total neutron structure factor of a trajectory.

    All H are transmuted to D, since the experimental reference is measured on
    the deuterated liquid. S(q) is obtained from the Fourier transform of the
    pair distribution function, weighted by coherent scattering lengths.

    Parameters
    ----------
    traj_path
        Path to the NPT trajectory of this model.
    work_dir
        Directory to write the converted trajectory and MDANSE output to.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Scattering vector in 1/A, and total structure factor.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    mdt_path = work_dir / "traj.mdt"
    ssf_prefix = work_dir / "ssf"

    # Cell fluctuates in NPT, so the cutoff must fit within every frame analysed
    cell_lengths = [
        float(atoms.cell.lengths().min()) for atoms in iread(traj_path, index=":")
    ]
    n_frames = len(cell_lengths)
    first_frame = int(n_frames * PRODUCTION_FRACTION)

    n_production = len(range(first_frame, n_frames, STRIDE))
    if n_production < MIN_FRAMES:
        raise ValueError(f"Only {n_production} production frames in {traj_path}")

    min_length = min(cell_lengths[first_frame:])
    if 2 * R_MAX_NM * NM_TO_ANG > min_length:
        raise ValueError(
            f"Cutoff {R_MAX_NM * NM_TO_ANG} A exceeds half the smallest cell "
            f"({min_length:.3f} A) in {traj_path}"
        )

    converter = Converter.create("ASE")
    converter.run(
        {
            "trajectory_file": str(traj_path),
            "atom_aliases": "{}",
            "time_step": FRAME_SPACING_FS,
            "time_unit": "fs",
            "n_steps": 0,
            "fold": True,
            "output_files": (str(work_dir / "traj"), 64, 128, "none", "no logs"),
        },
        status=False,
    )

    # Transmute H to D, matching the deuterated experimental sample
    trajectory = Trajectory(str(mdt_path))
    transmutation = {
        str(index): "H2"
        for index, atom_type in enumerate(trajectory.atom_types)
        if atom_type == "H"
    }
    trajectory.close()

    ssf = IJob.create("StaticStructureFactor")
    ssf.run(
        {
            "trajectory": str(mdt_path),
            "frames": (first_frame, n_frames, STRIDE),
            "r_values": (0.0, R_MAX_NM, R_STEP_NM),
            "q_values": (Q_MIN_INV_NM, Q_MAX_INV_NM, Q_STEP_INV_NM),
            "atom_selection": "{}",
            "atom_transmutation": json.dumps(transmutation),
            "grouping_level": "atom",
            "weights": "b_coherent",
            "output_files": (str(ssf_prefix), ["MDAFormat"], "no logs"),
            "running_mode": ("single-core", 1),
        },
        status=False,
    )

    with h5py.File(f"{ssf_prefix}.mda", "r") as output:
        q = output["ssf/axes/q"][:] * INV_NM_TO_INV_ANG
        sq = output["ssf/total"][:]

    return q, sq


def compute_r_factor(
    q_ref: np.ndarray,
    sq_ref: np.ndarray,
    q_calc: np.ndarray,
    sq_calc: np.ndarray,
) -> float:
    """
    Compute the R-factor between calculated and experimental S(q).

    The R-factor is ``sum|S_exp - S_calc| / sum|S_exp|``, evaluated on the
    experimental grid up to `SCORE_Q_MAX`.

    Parameters
    ----------
    q_ref
        Experimental scattering vector in 1/A.
    sq_ref
        Experimental structure factor.
    q_calc
        Calculated scattering vector in 1/A.
    sq_calc
        Calculated structure factor.

    Returns
    -------
    float
        R-factor, or NaN if the curves do not overlap.
    """
    mask = (q_ref >= q_calc.min()) & (q_ref <= min(SCORE_Q_MAX, q_calc.max()))
    if not mask.any():
        return np.nan

    # Identity when the calculated and experimental grids coincide
    sq_interp = np.interp(q_ref[mask], q_calc, sq_calc)

    return float(
        np.sum(np.abs(sq_ref[mask] - sq_interp)) / np.sum(np.abs(sq_ref[mask]))
    )


def first_peak_position(
    q: np.ndarray, sq: np.ndarray, q_min: float, q_max: float
) -> float:
    """
    Get the position of the maximum of S(q) over the scored range.

    The position is read directly off the grid of scattering vectors, which is
    fine enough that interpolating between points is not necessary.

    Parameters
    ----------
    q
        Scattering vector in 1/A.
    sq
        Structure factor.
    q_min
        Lower bound of the range searched, in 1/A.
    q_max
        Upper bound of the range searched, in 1/A.

    Returns
    -------
    float
        Position of the peak in 1/A, or NaN if the range contains no points.
    """
    (indices,) = np.nonzero((q >= q_min) & (q <= q_max) & ~np.isnan(sq))
    if indices.size == 0:
        return np.nan

    return float(q[indices[np.argmax(sq[indices])]])


@pytest.fixture
@plot_scatter(
    filename=OUT_PATH / "figure_sq.json",
    title="HF Neutron Structure Factor",
    x_label="q / 1/A",
    y_label="S(q)",
    show_line=True,
    show_markers=False,
    highlight_range={"Scored": [Q_MIN_INV_NM * INV_NM_TO_INV_ANG, SCORE_Q_MAX]},
)
def sq_curves() -> dict[str, list]:
    """
    Get experimental and predicted structure factors for all models.

    Returns
    -------
    dict[str, list]
        Scattering vectors and structure factors for the reference and all
        models with a trajectory.
    """
    q_ref, sq_ref = load_reference_sq()
    results = {"ref": [q_ref.tolist(), sq_ref.tolist()]}

    for model_name in MODELS:
        traj_path = CALC_PATH / model_name / "NPT.traj"

        # Missing models are left out of the plot, and scored as None
        if not traj_path.exists():
            continue

        try:
            q, sq = compute_sq(traj_path, CALC_PATH / model_name / "sq_mdanse")
        except Exception as exc:
            warn(
                f"Error computing structure factor for {model_name}: {exc}",
                stacklevel=2,
            )
            continue

        results[model_name] = [q.tolist(), sq.tolist()]

        sq_out = OUT_PATH / model_name
        sq_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            sq_out / "sq.dat",
            np.column_stack([q, sq]),
            header="q/1/A S(q)",
            fmt="%.6f",
        )

    return results


@pytest.fixture
def sq_errors(sq_curves: dict[str, list]) -> dict[str, dict]:
    """
    Get structure factor errors for all models.

    Parameters
    ----------
    sq_curves
        Scattering vectors and structure factors for the reference and models.

    Returns
    -------
    dict[str, dict]
        R-factors and first peak position errors for all models.
    """
    q_ref = np.array(sq_curves["ref"][0], dtype=float)
    sq_ref = np.array(sq_curves["ref"][1], dtype=float)

    # Peaks are located over the same range as the R-factor is evaluated on
    q_min = float(q_ref.min())
    peak_ref = first_peak_position(q_ref, sq_ref, q_min, SCORE_Q_MAX)

    r_factors = {}
    peak_errors = {}

    for model_name in MODELS:
        # Models without a structure factor are scored as None, as in `mae`
        if model_name not in sq_curves:
            r_factors[model_name] = None
            peak_errors[model_name] = None
            continue

        q = np.array(sq_curves[model_name][0], dtype=float)
        sq = np.array(sq_curves[model_name][1], dtype=float)

        r_factors[model_name] = compute_r_factor(q_ref, sq_ref, q, sq)
        peak_errors[model_name] = abs(
            first_peak_position(q, sq, q_min, SCORE_Q_MAX) - peak_ref
        )

    return {
        "S(q) R-factor": r_factors,
        "First Peak Position Error": peak_errors,
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "hf_structure_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_MODEL_NAMES,
    weights=DEFAULT_WEIGHTS,
)
def metrics(sq_errors: dict[str, dict]) -> dict[str, dict]:
    """
    Get all HF structure factor metrics.

    Parameters
    ----------
    sq_errors
        R-factors and first peak position errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return sq_errors


def test_hf_structure(metrics: dict[str, dict]) -> None:
    """
    Run HF structure factor test.

    Parameters
    ----------
    metrics
        All HF structure factor metrics.
    """
    write_struct_info(
        data_path=CALC_PATH / "mock" / "minimised.xyz",
        out_path=OUT_PATH,
        index=0,
    )
