"""Run phonon dispersion and thermal calculations for diamond."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
from typing import Any
from warnings import warn

from ase.constraints import FixSymmetry
from ase.io import write
from ase.optimize import FIRE
import numpy as np
from phonopy import load as load_phonopy
import pytest

from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import (
    get_fc2_and_freqs,
    init_phonopy_from_ref,
    phonopy_to_ase_atoms,
    qpath_distances,
)
from ml_peg.calcs.bulk_crystal.phonons.thermal_utils import compute_thermal_properties
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

GITHUB_BASE = "https://raw.githubusercontent.com/7radians/ml-peg-data/main"

OUT_PATH = Path(__file__).parent / "outputs"
DFT_REF_PATH = OUT_PATH / "DFT"

# Relaxation settings, matching the general phonon benchmark.
FMAX = 0.005
RELAX_STEPS = 1000

Q_MESH = 6
THERMAL_MESH = [20, 20, 20]
# 4×4×4 of the 8-atom conventional cell = 512-atom supercell.
GRUNEISEN_SUPERCELL = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]

# FCC diamond BZ path used by the CASTEP reference: Γ→X→W→K→Γ→L→U→W→L→K→X
HS_LABELS = [r"$\Gamma$", "X", "W", "K", r"$\Gamma$", "L", "U", "W", "L", "K", "X"]

MODELS = load_models(current_models)


@pytest.fixture(scope="session")
def diamond_data() -> Path:
    """
    Download diamond benchmark inputs and return the data directory.

    Returns
    -------
    Path
        Directory containing ``diamond.yaml``, ``dft_band.npz``, and
        ``diamond_thermal_ref.json``.
    """
    extracted = Path(
        download_github_data(filename="diamond_data/data.zip", github_uri=GITHUB_BASE)
    )
    return extracted / "data"


def _detect_hs_boundaries(qpoints: np.ndarray) -> list[int]:
    """
    Return indices of high-symmetry points along a continuous q-path.

    Detects direction changes by comparing consecutive step directions.
    Always includes the first and last index.

    Parameters
    ----------
    qpoints
        Array of fractional q-point coordinates, shape ``(nq, 3)``.

    Returns
    -------
    list[int]
        Indices of high-symmetry points including the first and last.
    """
    dq = np.diff(qpoints, axis=0)
    dq_norm = np.linalg.norm(dq, axis=1)
    dq_unit = dq / (dq_norm[:, None] + 1e-12)
    cosang = np.sum(dq_unit[1:] * dq_unit[:-1], axis=1)
    turns = list(np.where(cosang < 0.95)[0] + 1)
    return [0] + turns + [len(qpoints) - 1]


def _reference_band_path(
    data_dir: Path,
) -> tuple[list[np.ndarray], np.ndarray, list[str], list[bool]]:
    """
    Load the DFT reference q-path, split into high-symmetry segments.

    Parameters
    ----------
    data_dir
        Directory containing ``dft_band.npz`` and ``diamond.yaml``.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray, list[str], list[bool]]
        Q-point segments, reference frequencies (THz, shape ``(nq, n_bands)``),
        labels, and phonopy-style path connections.
    """
    ref = np.load(data_dir / "dft_band.npz", allow_pickle=False)
    qpoints = np.asarray(ref["qpoints"], dtype=float)
    freqs_thz = np.asarray(ref["freqs_cm1"], dtype=float) / 33.35640951981521

    boundaries = _detect_hs_boundaries(qpoints)
    if len(boundaries) != len(HS_LABELS):
        # Fall back to a single unlabelled segment.
        boundaries = [0, len(qpoints) - 1]

    segments = [
        qpoints[start : stop + 1]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    labels = HS_LABELS if len(boundaries) == len(HS_LABELS) else [HS_LABELS[0], "X"]
    connections = [True] * (len(segments) - 1) + [False]
    return segments, freqs_thz, labels, connections


def test_diamond_phonons_ref(diamond_data: Path) -> None:
    """
    Convert the DFT reference data to the shared phonon output formats.

    Writes ``outputs/DFT/diamond_band_structure.npz`` (pickled band dict),
    ``diamond_thermal.json``, and ``diamond.xyz``, mirroring the layout used
    by the general phonon benchmark so analysis and app code can be shared.

    Parameters
    ----------
    diamond_data
        Directory containing the downloaded reference data.
    """
    DFT_REF_PATH.mkdir(parents=True, exist_ok=True)

    phonons_ref = load_phonopy(str(diamond_data / "diamond.yaml"))
    segments, freqs_thz, labels, connections = _reference_band_path(diamond_data)

    # Distances in the phonopy band-structure convention so that reference and
    # model bands share the same x-axis in the app.
    qpoints_all = np.concatenate(
        [seg if i == 0 else seg[1:] for i, seg in enumerate(segments)]
    )
    distances = qpath_distances(qpoints_all, np.array(phonons_ref.primitive.cell))

    band_dict = {"distances": [], "frequencies": [], "labels": labels}
    band_dict["path_connections"] = connections
    start = 0
    for segment in segments:
        stop = start + len(segment) - 1
        band_dict["distances"].append(distances[start : stop + 1])
        band_dict["frequencies"].append(freqs_thz[start : stop + 1])
        start = stop

    with open(DFT_REF_PATH / "diamond_band_structure.npz", "wb") as handle:
        pickle.dump(band_dict, handle)

    shutil.copy2(
        diamond_data / "diamond_thermal_ref.json",
        DFT_REF_PATH / "diamond_thermal.json",
    )
    write(DFT_REF_PATH / "diamond.xyz", phonopy_to_ase_atoms(phonons_ref))


@pytest.mark.parametrize("mlip", MODELS.items())
def test_diamond_phonons(mlip: tuple[str, Any], diamond_data: Path) -> None:
    """
    Compute phonon band structure and thermal properties for diamond with one MLIP.

    MLIP: 512-atom conventional supercell for both band and thermal calculations.
    DFT ref: 128-atom primitive supercell (CASTEP/RSCAN).

    Parameters
    ----------
    mlip
        Tuple of (model_name, model) as provided by pytest parametrize.
    diamond_data
        Directory containing the downloaded reference data.
    """
    model_name, model = mlip
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    band_path = out_dir / "diamond_band_structure.npz"
    thermal_path = out_dir / "diamond_thermal.json"
    struct_path = out_dir / "diamond.xyz"

    if band_path.exists() and thermal_path.exists() and struct_path.exists():
        return

    calc = model.get_calculator(precision="high")
    phonons_ref = load_phonopy(str(diamond_data / "diamond.yaml"))
    segments, _, labels, connections = _reference_band_path(diamond_data)

    # Relax with fixed symmetry, as in the general phonon benchmark. The cell
    # is kept fixed so that band distances remain comparable to the reference.
    atoms = phonopy_to_ase_atoms(phonons_ref)
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms))
    try:
        FIRE(atoms, logfile=None).run(fmax=FMAX, steps=RELAX_STEPS)
    except Exception as exc:
        warn(f"{model_name}: diamond relaxation failed: {exc}", stacklevel=2)
        return
    atoms.set_constraint()
    atoms.calc = None
    write(struct_path, atoms)

    if not band_path.exists():
        try:
            phonons = init_phonopy_from_ref(
                atoms=atoms,
                displacement_dataset=phonons_ref.dataset,
                primitive_matrix=phonons_ref.primitive_matrix,
                symprec=1e-5,
            )
            phonons, _, _ = get_fc2_and_freqs(
                phonons=phonons,
                calculator=calc,
                q_mesh=np.array([Q_MESH] * 3),
                symmetrize_fc2=True,
            )
            phonons.run_band_structure(
                paths=segments, labels=labels, path_connections=connections
            )
            with open(band_path, "wb") as handle:
                pickle.dump(phonons.get_band_structure_dict(), handle)
        except Exception as exc:
            warn(f"{model_name}: diamond band structure failed: {exc}", stacklevel=2)

    if not thermal_path.exists():
        try:
            phonons_grun = init_phonopy_from_ref(
                atoms=atoms,
                fc2_supercell=GRUNEISEN_SUPERCELL,
                displacement_distance=0.01,
                is_plusminus=True,
            )
            phonons_grun, _, _ = get_fc2_and_freqs(
                phonons=phonons_grun,
                calculator=calc,
                symmetrize_fc2=True,
            )
            thermal = compute_thermal_properties(
                phonons=phonons_grun,
                atoms=atoms,
                calculator=calc,
                q_mesh=THERMAL_MESH,
                symmetrize_fc2=True,
                temperature=300.0,
            )
            thermal_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "mean_gamma": thermal["mean_gamma"],
                        "debye_temperature_K": thermal["debye_temperature_K"],
                        "kappa_W_per_mK": thermal["kappa_W_per_mK"],
                        "temperature_K": 300.0,
                    },
                    indent=2,
                ),
                encoding="utf8",
            )
        except Exception as exc:
            warn(f"{model_name}: diamond thermal failed: {exc}", stacklevel=2)
