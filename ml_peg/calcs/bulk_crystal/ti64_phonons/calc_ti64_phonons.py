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
from ml_peg.calcs.bulk_crystal.phonons.thermal_utils import (
    EV_TO_KJMOL,
    gaussian_dos,
    harmonic_free_energy,
)
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
DISPLACEMENT = 0.02

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


def patch_ase_castep_reader_dummy_energy() -> None:
    """Patch ASE CASTEP reader once to tolerate missing energy keys."""
    import ase.io.castep.castep_reader as cr

    if getattr(cr._set_energy_and_free_energy, "_ml_peg_patched", False):
        return
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
            results.setdefault("energy", 0.0)
            results.setdefault("free_energy", 0.0)

    _safe_set_energy_and_free_energy._ml_peg_patched = True
    cr._set_energy_and_free_energy = _safe_set_energy_and_free_energy


class PhononFromCastepPDOS(PhononFromCastep):
    """
    Parse CASTEP q-point phonon output for DOS/thermodynamics.

    Parameters
    ----------
    castep_file
        Path to a CASTEP output file containing q-point frequencies and weights.

    Attributes
    ----------
    frequencies
        Phonon frequencies array with shape ``(nq, 3N)``.
    weights
        Q-point weights array with shape ``(nq,)``.
    """

    def __init__(self, castep_file: str) -> None:
        """
        Initialise from a CASTEP file and extract frequencies/weights.

        Parameters
        ----------
        castep_file
            Path to the CASTEP q-points output file.
        """
        atoms = ase.io.read(castep_file)
        self.number_of_branches = len(atoms) * 3
        self.filename = castep_file
        self.read_in_file()
        self.get_frequencies()
        self.get_weights()
        delattr(self, "filelines")

    def get_weights(self) -> None:
        """Extract q-point weights (the last number on each ``q-pt=`` line)."""
        float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?")
        weights = [
            float(float_re.findall(line)[-1])
            for line in self.filelines
            if "q-pt=" in line
        ]

        nq_freq = self.frequencies.shape[0]
        if len(weights) != nq_freq:
            raise ValueError(
                f"Parsed {len(weights)} weights but frequencies has "
                f"{nq_freq} q-points in {self.filename}."
            )
        self.weights = np.array(weights, dtype=float)


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

# The k-path is determined by the case-name lattice prefix (see _case_path),
# the supercell is GRID_222, and displacements default to DISPLACEMENT unless
# overridden per case.
CASES: list[dict[str, Any]] = [
    {
        "case_name": "hcp_Ti6AlV",
        "structure_file": "ti64_hcp_phonon.castep",
        "qpoints_file": "ti64_hcp_phonon_qpoints.castep",
    },
    {
        "case_name": "bcc_Ti6AlV",
        "structure_file": "ti64_bcc_phonon.castep",
        "qpoints_file": "ti64_bcc_phonon_qpoints.castep",
    },
    {
        "case_name": "hex_Ti8AlV",
        "structure_file": "ti64_hex_phonon.castep",
        "qpoints_file": "ti64_hex_phonon_qpoints.castep",
        "disp_dos": 0.01,
    },
    {
        "case_name": "hcp_Ti6Al2",
        "structure_file": "ti64_hcp_phonon_AlAl_qpath.castep",
        "qpoints_file": "ti64_hcp_phonon_AlAl_qpoints.castep",
    },
    {
        "case_name": "hcp_Ti6V2",
        "structure_file": "ti64_hcp_phonon_VV_qpath.castep",
        "qpoints_file": "ti64_hcp_phonon_VV_qpoints.castep",
    },
    {
        "case_name": "hcp_Ti7V",
        "structure_file": "Ti7V_hcp_phonon_qpath.castep",
        "qpoints_file": "Ti7V_hcp_phonon_qpoints.castep",
    },
    {
        "case_name": "bcc_Ti6Al2",
        "structure_file": "ti64_bcc_phonon_AlAl_qpath.castep",
        "qpoints_file": "ti64_bcc_phonon_AlAl_qpoints.castep",
    },
    {
        "case_name": "bcc_Ti6V2",
        "structure_file": "ti64_bcc_phonon_VV_qpath.castep",
        "qpoints_file": "ti64_bcc_phonon_VV_qpoints.castep",
    },
    {
        "case_name": "hex_Ti10Al2",
        "structure_file": "ti64_hex_phonon_AlAl_qpath.castep",
        "qpoints_file": "ti64_hex_phonon_AlAl_qpoints.castep",
    },
    {
        "case_name": "hex_Ti10V2",
        "structure_file": "ti64_hex_phonon_VV_qpath.castep",
        "qpoints_file": "ti64_hex_phonon_VV_qpoints.castep",
    },
]


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
    kpath, labels = _case_path(case)

    atoms = ase.io.read(structure_file)
    atoms.calc = None

    # Dispersion along the k-path, split into high-symmetry segments with
    # distances in the phonopy convention so reference and model bands share
    # the same x-axis.
    pfc = PhononFromCastep(castep_file=str(structure_file), kpath_in=kpath)
    distances = qpath_distances(pfc.kpath, atoms.cell.array)

    band_dict: dict[str, Any] = {
        "distances": [],
        "frequencies": [],
        "labels": labels,
        "path_connections": [True] * (len(pfc.kpath_idx) - 1) + [False],
    }
    for indices in pfc.kpath_idx:
        band_dict["distances"].append(distances[indices])
        band_dict["frequencies"].append(np.asarray(pfc.frequencies)[indices])

    with open(DFT_REF_PATH / f"{case}_band_structure.npz", "wb") as handle:
        pickle.dump(band_dict, handle)

    # DOS and free energy from the CASTEP q-point frequencies and weights.
    pfc_q = PhononFromCastepPDOS(str(data_dir / spec["qpoints_file"]))
    q_freqs = np.asarray(pfc_q.frequencies, dtype=float)
    q_weights = np.asarray(pfc_q.weights, dtype=float)

    grid = np.arange(
        q_freqs.min() - 0.5, q_freqs.max() + 0.5, DOS_SIGMA_THZ / 5, dtype=float
    )
    dos_dict = {
        "frequency_points": grid,
        "total_dos": gaussian_dos(q_freqs, q_weights, grid, DOS_SIGMA_THZ),
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
    kpath, labels = _case_path(case)
    disp_dos = spec.get("disp_dos", DISPLACEMENT)

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
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [spec for spec in CASES if not _case_complete(spec["case_name"], out_dir)]
    if not pending:
        return

    calc = model.get_calculator(precision="high")

    for spec in tqdm(pending, desc=f"{model_name} Ti64 phonons", unit="case"):
        try:
            _calc_case(spec, calc, ti64_data, out_dir)
        except Exception as exc:
            warn(
                f"{model_name}: Ti64 case {spec['case_name']} failed: {exc}",
                stacklevel=2,
            )
