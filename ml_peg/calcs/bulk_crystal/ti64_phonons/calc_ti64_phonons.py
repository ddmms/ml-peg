"""
Run Ti64 phonon suite.

Outputs use the same per-system file formats as the general phonon benchmark
(``<case>_band_structure.npz``, ``<case>_dos.npz``,
``<case>_thermal_properties.json``, ``<case>.xyz``). The CASTEP (PBE)
reference data is downloaded pre-converted to the same formats and copied to
``outputs/DFT/``.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
from typing import Any
from warnings import warn

from ase.constraints import FixSymmetry
import ase.io
from ase.optimize import FIRE
import numpy as np
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import pytest
from tqdm import tqdm

from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import (
    get_fc2_and_freqs,
    init_phonopy_from_ref,
)
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

GITHUB_BASE = "https://raw.githubusercontent.com/7radians/ml-peg-data/main"

OUT_PATH = CALCS_ROOT / "bulk_crystal" / "ti64_phonons" / "outputs"
DFT_REF_PATH = OUT_PATH / "DFT"

# Relaxation settings: FIRE with fixed symmetry, as in the general phonon
# benchmark. fmax is tighter as the cells are small.
FMAX = 0.001
RELAX_STEPS = 1000

Q_MESH_THERMAL = [20, 20, 20]
KPOINTS = 100
T_MIN, T_MAX, T_STEP = 0, 2000, 10
DISPLACEMENT = 0.02
# Per-case DOS displacement overrides; all other settings are shared.
DISP_DOS = {"hex_Ti8AlV": 0.01}

CASES = [
    "hcp_Ti6AlV",
    "bcc_Ti6AlV",
    "hex_Ti8AlV",
    "hcp_Ti6Al2",
    "hcp_Ti6V2",
    "hcp_Ti7V",
    "bcc_Ti6Al2",
    "bcc_Ti6V2",
    "hex_Ti10Al2",
    "hex_Ti10V2",
]

MODELS = load_models(current_models)


@pytest.fixture(scope="session")
def ti64_data() -> Path:
    """
    Download Ti64 benchmark reference data and return the data directory.

    Returns
    -------
    Path
        Directory containing the pre-converted CASTEP reference data
        (band structures, DOS, free energies, and structures per case).
    """
    extracted = Path(
        download_github_data(filename="ti64_data/data.zip", github_uri=GITHUB_BASE)
    )
    return extracted / "data"


def _hex_path() -> tuple[list[list[float]], list[str]]:
    """
    Return the high-symmetry path and tick labels for the hexagonal cell.

    Returns
    -------
    tuple[list[list[float]], list[str]]
        ``(kpath, labels)`` for the hexagonal BZ.
    """
    gam = [0, 0, 0]
    a_pt = [0, 0, 1 / 2]
    k_pt = [1 / 3, 1 / 3, 0]
    m_pt = [0.5, 0, 0]
    return [gam, k_pt, m_pt, gam, a_pt], ["$\\Gamma$", "K", "M", "$\\Gamma$", "A"]


def _bcc_path() -> tuple[list[list[float]], list[str]]:
    """
    Return the high-symmetry path and tick labels for the BCC cell.

    Returns
    -------
    tuple[list[list[float]], list[str]]
        ``(kpath, labels)`` for the BCC BZ.
    """
    gam = [0, 0, 0]
    h_pt = [0.5, -0.5, 0.5]
    p_pt = [0.25, 0.25, 0.25]
    n_pt = [0, 0, 0.5]
    return (
        [gam, h_pt, n_pt, gam, p_pt, h_pt, p_pt, n_pt],
        ["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H", "P", "N"],
    )


GRID_222 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
HEX_KPATH, HEX_LABELS = _hex_path()
BCC_KPATH, BCC_LABELS = _bcc_path()


def _case_path(case_name: str) -> tuple[list[list[float]], list[str]]:
    """
    Return the k-path and labels for a case from its lattice prefix.

    Parameters
    ----------
    case_name
        Case identifier starting with ``bcc``, ``hcp``, or ``hex``.

    Returns
    -------
    tuple[list[list[float]], list[str]]
        ``(kpath, labels)`` for the case's Brillouin zone.
    """
    if case_name.startswith("bcc"):
        return BCC_KPATH, BCC_LABELS
    return HEX_KPATH, HEX_LABELS


def test_ti64_phonons_ref(ti64_data: Path) -> None:
    """
    Copy the pre-converted CASTEP reference data to ``outputs/DFT/``.

    Parameters
    ----------
    ti64_data
        Directory containing the downloaded reference data.
    """
    DFT_REF_PATH.mkdir(parents=True, exist_ok=True)
    for src in sorted(ti64_data.iterdir()):
        shutil.copy2(src, DFT_REF_PATH / src.name)


def _case_complete(case: str, out_dir: Path, do_thermal: bool) -> bool:
    """
    Return True when all model outputs for a case already exist.

    Parameters
    ----------
    case
        Case identifier.
    out_dir
        Directory containing model outputs.
    do_thermal
        Whether thermal properties are expected for this case.

    Returns
    -------
    bool
        Whether all expected output files are present.
    """
    names = [f"{case}_band_structure.npz", f"{case}_dos.npz", f"{case}.xyz"]
    if do_thermal:
        names.append(f"{case}_thermal_properties.json")
    return all((out_dir / name).exists() for name in names)


def _calc_case(case: str, calc: Any, data_dir: Path, out_dir: Path) -> None:
    """
    Run one Ti64 phonon case for one model and write outputs.

    Thermal properties are computed for the cases with a free-energy
    reference in the downloaded data.

    Parameters
    ----------
    case
        Case identifier.
    calc
        ASE calculator for the model under test.
    data_dir
        Directory containing the pre-converted reference data.
    out_dir
        Directory where model outputs are written.
    """
    kpath, labels = _case_path(case)
    disp_dos = DISP_DOS.get(case, DISPLACEMENT)

    # Relax with fixed symmetry (positions only; the cell is kept fixed so
    # band distances remain comparable to the reference).
    atoms = ase.io.read(data_dir / f"{case}.xyz")
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms))
    FIRE(atoms, logfile=None).run(fmax=FMAX, steps=RELAX_STEPS)
    atoms.set_constraint()
    atoms.calc = None
    ase.io.write(out_dir / f"{case}.xyz", atoms)

    # Dispersion
    phonons = init_phonopy_from_ref(
        atoms=atoms,
        fc2_supercell=GRID_222,
        primitive_matrix=None,
        displacement_distance=DISPLACEMENT,
        is_plusminus=True,
    )
    phonons, _, _ = get_fc2_and_freqs(phonons, calc, symmetrize_fc2=True)
    qpts, conns = get_band_qpoints_and_path_connections([kpath], npoints=KPOINTS)
    phonons.run_band_structure(qpts, path_connections=conns, labels=labels)
    with open(out_dir / f"{case}_band_structure.npz", "wb") as handle:
        pickle.dump(phonons.get_band_structure_dict(), handle)

    # DOS (recompute force constants only if the displacement differs)
    if disp_dos != DISPLACEMENT:
        phonons_dos = init_phonopy_from_ref(
            atoms=atoms,
            fc2_supercell=GRID_222,
            primitive_matrix=None,
            displacement_distance=disp_dos,
            is_plusminus=True,
        )
        phonons_dos, _, _ = get_fc2_and_freqs(phonons_dos, calc, symmetrize_fc2=True)
    else:
        phonons_dos = phonons
    phonons_dos.run_mesh(Q_MESH_THERMAL)
    phonons_dos.run_total_dos()
    with open(out_dir / f"{case}_dos.npz", "wb") as handle:
        pickle.dump(phonons_dos.get_total_dos_dict(), handle)

    # Thermodynamics
    if (data_dir / f"{case}_thermal_properties.json").exists():
        phonons_dos.run_thermal_properties(t_min=T_MIN, t_max=T_MAX, t_step=T_STEP)
        thermal = phonons_dos.get_thermal_properties_dict()
        thermal_safe = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in thermal.items()
        }
        thermal_safe["n_atoms"] = len(atoms)
        with open(
            out_dir / f"{case}_thermal_properties.json", "w", encoding="utf8"
        ) as handle:
            json.dump(thermal_safe, handle, indent=4)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_ti64_phonons(mlip: tuple[str, Any], ti64_data: Path) -> None:
    """
    Run the full Ti64 phonon suite for one model and write outputs.

    Parameters
    ----------
    mlip
        Tuple of (model_name, model) as provided by pytest parametrize.
    ti64_data
        Directory containing the pre-converted reference data.
    """
    model_name, model = mlip
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        case
        for case in CASES
        if not _case_complete(
            case,
            out_dir,
            do_thermal=(ti64_data / f"{case}_thermal_properties.json").exists(),
        )
    ]
    if not pending:
        return

    calc = model.get_calculator(precision="high")

    for case in tqdm(pending, desc=f"{model_name} Ti64 phonons", unit="case"):
        try:
            _calc_case(case, calc, ti64_data, out_dir)
        except Exception as exc:
            warn(f"{model_name}: Ti64 case {case} failed: {exc}", stacklevel=2)
