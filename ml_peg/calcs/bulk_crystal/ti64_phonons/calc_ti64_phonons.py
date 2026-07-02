"""
Run Ti64 CASTEP phonon suite.

Outputs use the same per-system file formats as the general phonon benchmark
(``<case>_band_structure.npz``, ``<case>_dos.npz``,
``<case>_thermal_properties.json``, ``<case>.xyz``), with the CASTEP reference
converted to the same formats under ``outputs/DFT/``.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
from typing import Any
from warnings import warn

from ase.constraints import FixSymmetry
import ase.io
from ase.optimize import FIRE
from ase.units import kJ, mol
import numpy as np
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import pytest
from tqdm import tqdm

from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import (
    get_fc2_and_freqs,
    init_phonopy_from_ref,
    qpath_distances,
)
from ml_peg.calcs.bulk_crystal.phonons.thermal_utils import harmonic_free_energy
from ml_peg.calcs.utils.CASTEP_reader_phonon_dispersion import PhononFromCastep
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
DOS_SIGMA_THZ = 0.05
TEMPERATURES = np.arange(0, 2001, 10, dtype=float)
EV_TO_KJMOL = mol / kJ

MODELS = load_models(current_models)


@pytest.fixture(scope="session")
def ti64_data() -> Path:
    """
    Download Ti64 benchmark inputs and return the data directory.

    Returns
    -------
    Path
        Directory containing the CASTEP reference outputs.
    """
    extracted = Path(
        download_github_data(filename="ti64_data/data.zip", github_uri=GITHUB_BASE)
    )
    return extracted / "data"


def patch_ase_castep_reader_dummy_energy(dummy_energy: float = 0.0) -> None:
    """
    Patch ASE CASTEP reader to tolerate missing energy keys.

    Parameters
    ----------
    dummy_energy
        Dummy energy/free energy (eV) to use when missing.
    """
    import ase.io.castep.castep_reader as cr

    orig = cr._set_energy_and_free_energy

    def _safe_set_energy_and_free_energy(results: dict[str, Any]) -> None:
        """
        Set energy keys, tolerating missing values from CASTEP phonon outputs.

        Parameters
        ----------
        results
            ASE results dictionary to patch in-place.
        """
        try:
            orig(results)
        except KeyError:
            results.setdefault("energy", float(dummy_energy))
            results.setdefault("free_energy", float(dummy_energy))

    cr._set_energy_and_free_energy = _safe_set_energy_and_free_energy


class PhononFromCastepPDOS(PhononFromCastep):
    """
    Parse CASTEP q-point phonon output for DOS/thermodynamics.

    Parameters
    ----------
    castep_file
        Path to a CASTEP output file containing q-point frequencies and weights.
    atoms_in
        Structure used to determine the number of branches (3N).

    Attributes
    ----------
    frequencies
        Phonon frequencies array with shape ``(nq, 3N)``.
    weights
        Q-point weights array with shape ``(nq,)``.
    """

    def __init__(self, castep_file: str, atoms_in: Any | None = None) -> None:
        """
        Initialise from a CASTEP file and extract frequencies/weights.

        Parameters
        ----------
        castep_file
            Path to the CASTEP q-points output file.
        atoms_in
            Structure used to determine the number of branches (3N). Required.
        """
        if atoms_in is None:
            raise ValueError(
                "atoms_in must be provided to set number_of_branches (=3N)."
            )

        self.number_of_branches = len(atoms_in) * 3
        self.filename = castep_file
        self.read_in_file()
        self.get_frequencies()
        self.get_weights()
        delattr(self, "filelines")

    def get_weights(self) -> None:
        """Extract q-point weights from filelines."""
        float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?")
        weights: list[float] = []

        for line in self.filelines:
            if "q-pt" not in line:
                continue

            match = re.search(
                r"weight\s*=\s*(" + float_re.pattern + r")",
                line,
                flags=re.I,
            )
            if match:
                weights.append(float(match.group(1)))
                continue

            nums = float_re.findall(line)
            if nums:
                weights.append(float(nums[-1]))

        nq_freq = self.frequencies.shape[0]

        if not weights:
            self.weights = np.ones(nq_freq, dtype=float)
            return

        w = np.array(weights, dtype=float)
        if w.shape[0] > nq_freq:
            w = w[-nq_freq:]

        if w.shape[0] != nq_freq:
            raise ValueError(
                f"Parsed {w.shape[0]} weights but frequencies has {nq_freq} q-points."
            )

        self.weights = w


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

# Cases with thermodynamic (free energy) outputs.
TP_ON = {
    "hcp_Ti6AlV",
    "hex_Ti8AlV",
    "hcp_Ti6Al2",
    "hcp_Ti6V2",
    "hcp_Ti7V",
    "hex_Ti10Al2",
    "hex_Ti10V2",
}

CASES: list[dict[str, Any]] = [
    {
        "case_name": "hcp_Ti6AlV",
        "structure_file": "ti64_hcp_phonon.castep",
        "qpoints_file": "ti64_hcp_phonon_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
    {
        "case_name": "bcc_Ti6AlV",
        "structure_file": "ti64_bcc_phonon.castep",
        "qpoints_file": "ti64_bcc_phonon_qpoints.castep",
        "kpath": BCC_KPATH,
        "labels": BCC_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
    {
        "case_name": "hex_Ti8AlV",
        "structure_file": "ti64_hex_phonon.castep",
        "qpoints_file": "ti64_hex_phonon_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.01,
    },
    {
        "case_name": "hcp_Ti6Al2",
        "structure_file": "ti64_hcp_phonon_AlAl_qpath.castep",
        "qpoints_file": "ti64_hcp_phonon_AlAl_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
    {
        "case_name": "hcp_Ti6V2",
        "structure_file": "ti64_hcp_phonon_VV_qpath.castep",
        "qpoints_file": "ti64_hcp_phonon_VV_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
    {
        "case_name": "hcp_Ti7V",
        "structure_file": "Ti7V_hcp_phonon_qpath.castep",
        "qpoints_file": "Ti7V_hcp_phonon_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
    {
        "case_name": "bcc_Ti6Al2",
        "structure_file": "ti64_bcc_phonon_AlAl_qpath.castep",
        "qpoints_file": "ti64_bcc_phonon_AlAl_qpoints.castep",
        "kpath": BCC_KPATH,
        "labels": BCC_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
    {
        "case_name": "bcc_Ti6V2",
        "structure_file": "ti64_bcc_phonon_VV_qpath.castep",
        "qpoints_file": "ti64_bcc_phonon_VV_qpoints.castep",
        "kpath": BCC_KPATH,
        "labels": BCC_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
    {
        "case_name": "hex_Ti10Al2",
        "structure_file": "ti64_hex_phonon_AlAl_qpath.castep",
        "qpoints_file": "ti64_hex_phonon_AlAl_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
    {
        "case_name": "hex_Ti10V2",
        "structure_file": "ti64_hex_phonon_VV_qpath.castep",
        "qpoints_file": "ti64_hex_phonon_VV_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_dos": 0.02,
    },
]


def _gaussian_dos(
    frequencies: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian-broadened phonon DOS on a frequency grid.

    Parameters
    ----------
    frequencies
        Phonon frequencies (THz), shape ``(nq, n_bands)``.
    weights
        Q-point weights, shape ``(nq,)``.
    grid
        Frequency grid (THz) to evaluate the DOS on.
    sigma
        Gaussian broadening (THz).

    Returns
    -------
    np.ndarray
        DOS values on ``grid``.
    """
    freqs = np.asarray(frequencies, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float)[:, None], freqs.shape).ravel()
    freqs = freqs.ravel()
    diff = freqs[:, None] - np.asarray(grid, dtype=float)[None, :]
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return norm * np.sum(w[:, None] * np.exp(-0.5 * (diff / sigma) ** 2), axis=0)


def _write_ref_case(spec: dict[str, Any], data_dir: Path) -> None:
    """
    Write DFT reference outputs for one Ti64 case in the shared formats.

    Parameters
    ----------
    spec
        Case specification from ``CASES``.
    data_dir
        Directory containing the CASTEP reference outputs.
    """
    case = spec["case_name"]
    structure_file = data_dir / spec["structure_file"]

    atoms = ase.io.read(structure_file)
    atoms.calc = None

    # Dispersion along the k-path, split into high-symmetry segments with
    # distances in the phonopy convention so reference and model bands share
    # the same x-axis.
    pfc = PhononFromCastep(castep_file=str(structure_file), kpath_in=spec["kpath"])
    distances = qpath_distances(pfc.kpath, atoms.cell.array)

    band_dict: dict[str, Any] = {
        "distances": [],
        "frequencies": [],
        "labels": spec["labels"],
        "path_connections": [True] * (len(pfc.kpath_idx) - 1) + [False],
    }
    for indices in pfc.kpath_idx:
        band_dict["distances"].append(distances[indices])
        band_dict["frequencies"].append(np.asarray(pfc.frequencies)[indices])

    with open(DFT_REF_PATH / f"{case}_band_structure.npz", "wb") as handle:
        pickle.dump(band_dict, handle)

    # DOS and free energy from the CASTEP q-point frequencies and weights.
    qpoints_file = data_dir / spec["qpoints_file"]
    pfc_q = PhononFromCastepPDOS(
        castep_file=str(qpoints_file), atoms_in=ase.io.read(qpoints_file)
    )
    q_freqs = np.asarray(pfc_q.frequencies, dtype=float)
    q_weights = np.asarray(pfc_q.weights, dtype=float)

    grid = np.arange(
        q_freqs.min() - 0.5, q_freqs.max() + 0.5, DOS_SIGMA_THZ / 5, dtype=float
    )
    dos_dict = {
        "frequency_points": grid,
        "total_dos": _gaussian_dos(q_freqs, q_weights, grid, DOS_SIGMA_THZ),
    }
    with open(DFT_REF_PATH / f"{case}_dos.npz", "wb") as handle:
        pickle.dump(dos_dict, handle)

    if case in TP_ON:
        free_energy_ev = harmonic_free_energy(q_freqs, q_weights, TEMPERATURES)
        thermal = {
            "temperatures": TEMPERATURES.tolist(),
            # kJ/mol (per cell), matching phonopy's thermal properties units.
            "free_energy": (free_energy_ev * EV_TO_KJMOL).tolist(),
            "n_atoms": len(atoms),
        }
        with open(
            DFT_REF_PATH / f"{case}_thermal_properties.json", "w", encoding="utf8"
        ) as handle:
            json.dump(thermal, handle, indent=4)

    ase.io.write(DFT_REF_PATH / f"{case}.xyz", atoms)


def test_ti64_phonons_ref(ti64_data: Path) -> None:
    """
    Convert the CASTEP reference data to the shared phonon output formats.

    Parameters
    ----------
    ti64_data
        Directory containing the downloaded CASTEP reference outputs.
    """
    patch_ase_castep_reader_dummy_energy()
    DFT_REF_PATH.mkdir(parents=True, exist_ok=True)

    for spec in tqdm(CASES, desc="Ti64 DFT reference"):
        _write_ref_case(spec, ti64_data)


def _case_complete(case: str, out_dir: Path) -> bool:
    """
    Return True when all model outputs for a case already exist.

    Parameters
    ----------
    case
        Case identifier.
    out_dir
        Directory containing model outputs.

    Returns
    -------
    bool
        Whether all expected output files are present.
    """
    names = [f"{case}_band_structure.npz", f"{case}_dos.npz", f"{case}.xyz"]
    if case in TP_ON:
        names.append(f"{case}_thermal_properties.json")
    return all((out_dir / name).exists() for name in names)


def _calc_case(spec: dict[str, Any], calc: Any, data_dir: Path, out_dir: Path) -> None:
    """
    Run one Ti64 phonon case for one model and write outputs.

    Parameters
    ----------
    spec
        Case specification from ``CASES``.
    calc
        ASE calculator for the model under test.
    data_dir
        Directory containing the CASTEP reference outputs.
    out_dir
        Directory where model outputs are written.
    """
    case = spec["case_name"]

    # Relax with fixed symmetry (positions only; the cell is kept fixed so
    # band distances remain comparable to the reference).
    atoms = ase.io.read(data_dir / spec["structure_file"])
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
        fc2_supercell=spec["grid"],
        primitive_matrix=None,
        displacement_distance=spec["disp_phonons"],
        is_plusminus=True,
    )
    phonons, _, _ = get_fc2_and_freqs(phonons, calc, symmetrize_fc2=True)
    qpts, conns = get_band_qpoints_and_path_connections(
        [spec["kpath"]], npoints=KPOINTS
    )
    phonons.run_band_structure(qpts, path_connections=conns, labels=spec["labels"])
    with open(out_dir / f"{case}_band_structure.npz", "wb") as handle:
        pickle.dump(phonons.get_band_structure_dict(), handle)

    # DOS (recompute force constants only if the displacement differs)
    if spec["disp_dos"] != spec["disp_phonons"]:
        phonons_dos = init_phonopy_from_ref(
            atoms=atoms,
            fc2_supercell=spec["grid"],
            primitive_matrix=None,
            displacement_distance=spec["disp_dos"],
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
    if case in TP_ON:
        phonons_dos.run_thermal_properties(
            t_min=TEMPERATURES[0], t_max=TEMPERATURES[-1], t_step=10
        )
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
        Directory containing the downloaded CASTEP reference outputs.
    """
    patch_ase_castep_reader_dummy_energy()

    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in tqdm(CASES, desc=f"{model_name} Ti64 phonons", unit="case"):
        if _case_complete(spec["case_name"], out_dir):
            continue
        try:
            _calc_case(spec, calc, ti64_data, out_dir)
        except Exception as exc:
            warn(
                f"{model_name}: Ti64 case {spec['case_name']} failed: {exc}",
                stacklevel=2,
            )
