"""
Run Ti64 CASTEP phonon suite (raw outputs only).

This module writes per-case raw calculation outputs for each MLIP model to
``outputs/<model_name>/`` as:

- ``<case>.npz``: raw arrays used by analysis
- ``<case>.json``: minimal metadata (no metrics)
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import ase.io
from ase.optimize import LBFGS
import numpy as np
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms
import pytest

from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import get_fc2_and_freqs
from ml_peg.calcs.utils.CASTEP_reader_phonon_dispersion import PhononFromCastep
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names, load_models

GITHUB_BASE = "https://raw.githubusercontent.com/7radians/ml-peg-data/main"

EXTRACTED_ROOT = Path(
    download_github_data(
        filename="ti64_data/data.zip",
        github_uri=GITHUB_BASE,
    )
)

DATA_PATH = EXTRACTED_ROOT / "data"

OUT_PATH = CALCS_ROOT / "bulk_crystal" / "ti64_phonons" / "outputs"

FMAX = 0.001
STEPS = 10000
MESH_202020 = [20, 20, 20]
KPOINTS = 100


def _json_default(obj: Any) -> Any:
    """
    JSON serializer for numpy/Path objects.

    Parameters
    ----------
    obj
        Object to convert.

    Returns
    -------
    Any
        JSON-serialisable representation.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_case_npz(case_name: str, model_name: str, **arrays: Any) -> None:
    """
    Write raw per-case arrays to a compressed NPZ file.

    Parameters
    ----------
    case_name
        Case identifier used in the output filename.
    model_name
        Model identifier used in the output directory.
    **arrays
        Keyword arrays to store in the NPZ file.
    """
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{case_name}.npz", **arrays)


def write_case_metadata(case_name: str, model_name: str, meta: dict[str, Any]) -> None:
    """
    Write minimal per-case metadata to JSON (no metrics).

    Parameters
    ----------
    case_name
        Case identifier used in the output filename.
    model_name
        Model identifier used in the output directory.
    meta
        Metadata mapping to write.
    """
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = {"case": case_name, "model_name": model_name, **meta}
    out_file = out_dir / f"{case_name}.json"
    with out_file.open("w", encoding="utf8") as handle:
        json.dump(meta_out, handle, indent=2, default=_json_default)


def patch_ase_castep_reader_dummy_energy(dummy_energy: float = 0.0) -> None:
    """
    Patch ASE CASTEP reader to tolerate missing energy keys.

    Parameters
    ----------
    dummy_energy
        Dummy energy/free energy (eV) to use when missing.
    """
    import ase.io.castep.castep_reader as cr  # noqa: WPS433 (local import is intentional)

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
    Parse CASTEP q-point phonon output for PDOS.

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


def run_case(
    *,
    case_name: str,
    structure_file: str,
    qpoints_file: str | None,
    kpath: list,
    labels: list,
    grid: list,
    disp_phonons: float,
    disp_pdos: float,
    do_tp: bool,
    calc: Any,
    model_name: str,
) -> None:
    """
    Run one Ti64 phonon case and write raw artifacts.

    Parameters
    ----------
    case_name
        Case identifier.
    structure_file
        CASTEP structure file path.
    qpoints_file
        Optional CASTEP qpoints file path.
    kpath
        High-symmetry k-path used by CASTEP dispersion output.
    labels
        Tick labels for the k-path.
    grid
        Phonon supercell grid (3x3 matrix-like list).
    disp_phonons
        Displacement magnitude for dispersion calculation.
    disp_pdos
        Displacement magnitude for DOS/PDOS calculation.
    do_tp
        Whether to compute thermo/TP outputs.
    calc
        ASE calculator from the selected model.
    model_name
        Model identifier used for output paths.
    """
    print(f"{case_name} | {model_name}")

    # DFT dispersion along k-path (raw reference)
    pfc = PhononFromCastep(castep_file=structure_file, kpath_in=kpath)

    # Relax
    atoms = ase.io.read(structure_file)
    atoms.calc = calc
    LBFGS(atoms, logfile=None).run(fmax=FMAX, steps=STEPS)

    # Build phonopy unitcell from relaxed atoms
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions(),
    )

    # ML dispersion (plus/minus displacements)
    ph_disp = Phonopy(unitcell, grid)
    ph_disp.generate_displacements(distance=disp_phonons, is_plusminus=True)
    ph_disp, _, _ = get_fc2_and_freqs(ph_disp, calc, symmetrize_fc2=False)
    qpts, conns = get_band_qpoints_and_path_connections([kpath], npoints=KPOINTS)
    ph_disp.run_band_structure(qpts, with_eigenvectors=True, path_connections=conns)
    bs = ph_disp.get_band_structure_dict()
    ml_frequencies = np.vstack(bs["frequencies"])
    normal_ticks = np.array([i * KPOINTS for i in range(len(kpath))], dtype=float)

    # ML DOS/PDOS
    ph_pdos = Phonopy(unitcell, grid)
    ph_pdos.generate_displacements(distance=disp_pdos)
    ph_pdos, _, _ = get_fc2_and_freqs(ph_pdos, calc, symmetrize_fc2=False)
    ph_pdos.run_mesh(MESH_202020, with_eigenvectors=True, is_mesh_symmetry=False)
    ph_pdos.run_projected_dos()
    ph_pdos.run_total_dos()
    pdos = ph_pdos.get_projected_dos_dict()
    dos = ph_pdos.get_total_dos_dict()

    # ML thermo
    tp_dict: dict[str, Any] = {}
    if do_tp:
        ph_pdos.run_mesh(MESH_202020)
        ph_pdos.run_thermal_properties(t_step=1, t_max=2000, t_min=0)
        tp_dict = ph_pdos.get_thermal_properties_dict()

    # DFT qpoints
    q_weights: np.ndarray | None = None
    q_freq_dft: np.ndarray | None = None
    if qpoints_file is not None:
        q_atoms = ase.io.read(qpoints_file)
        pfc_q = PhononFromCastepPDOS(castep_file=qpoints_file, atoms_in=q_atoms)
        q_weights = np.asarray(pfc_q.weights, dtype=float)
        q_freq_dft = np.asarray(pfc_q.frequencies, dtype=float)

    # JSON metadata
    meta: dict[str, Any] = {
        "structure_file": str(structure_file),
        "qpoints_file": str(qpoints_file) if qpoints_file is not None else None,
        "kpath": [kpath],
        "labels": labels,
        "phonon_grid": grid,
        "displacement_phonons": float(disp_phonons),
        "displacement_pdos": float(disp_pdos),
        "relax_settings": {"fmax": float(FMAX), "steps": int(STEPS)},
        "did_tp": bool(do_tp),
        "tp_settings": {"mesh": MESH_202020, "tmax": 2000, "tstep": 1}
        if do_tp
        else None,
    }
    write_case_metadata(case_name, model_name, meta)

    arrays: dict[str, np.ndarray] = {
        "n_atoms": np.int64(len(atoms)),
        "labels": np.array(labels, dtype=object),
        "kpath": np.array([kpath], dtype=object),
        "phonon_grid": np.asarray(grid, dtype=int),
        # DFT dispersion
        "dft_x": np.asarray(pfc.xscale, dtype=float),
        "dft_frequencies": np.asarray(pfc.frequencies, dtype=float),
        # ML dispersion
        "ml_frequencies": ml_frequencies,
        "ml_normal_ticks": normal_ticks,
    }

    # PDOS
    if "frequency_points" in pdos:
        arrays["pdos_frequency_points"] = np.asarray(
            pdos["frequency_points"], dtype=float
        )
    if "projected_dos" in pdos:
        arrays["pdos_projected"] = np.asarray(pdos["projected_dos"], dtype=float)
    if "dos" in dos:
        arrays["dos_total"] = np.asarray(dos["dos"], dtype=float)

    # TP
    if do_tp and tp_dict:
        if "temperatures" in tp_dict:
            arrays["tp_temperatures"] = np.asarray(tp_dict["temperatures"], dtype=float)
        if "free_energy" in tp_dict:
            arrays["tp_free_energy"] = np.asarray(tp_dict["free_energy"], dtype=float)

    # DFT qpoints
    if qpoints_file is not None:
        arrays["q_weights"] = np.asarray(q_weights, dtype=float)  # type: ignore[arg-type]
        arrays["q_frequencies_dft"] = np.asarray(q_freq_dft, dtype=float)  # type: ignore[arg-type]

    write_case_npz(case_name, model_name, **arrays)


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

TP_ON = {
    "hcp_Ti6AlV",  # 1/10
    "hex_Ti8AlV",  # 3/10
    "hcp_Ti6Al2",  # 4/10
    "hcp_Ti6V2",  # 5/10
    "hcp_Ti7V",  # 6/10
    "hex_Ti10Al2",  # 9/10
    "hex_Ti10V2",  # 10/10
}

CASES: list[dict[str, Any]] = [
    {
        "case_name": "hcp_Ti6AlV",
        "structure_file": DATA_PATH / "ti64_hcp_phonon.castep",
        "qpoints_file": DATA_PATH / "ti64_hcp_phonon_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
    {
        "case_name": "bcc_Ti6AlV",
        "structure_file": DATA_PATH / "ti64_bcc_phonon.castep",
        "qpoints_file": DATA_PATH / "ti64_bcc_phonon_qpoints.castep",
        "kpath": BCC_KPATH,
        "labels": BCC_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
    {
        "case_name": "hex_Ti8AlV",
        "structure_file": DATA_PATH / "ti64_hex_phonon.castep",
        "qpoints_file": DATA_PATH / "ti64_hex_phonon_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.01,
    },
    {
        "case_name": "hcp_Ti6Al2",
        "structure_file": DATA_PATH / "ti64_hcp_phonon_AlAl_qpath.castep",
        "qpoints_file": DATA_PATH / "ti64_hcp_phonon_AlAl_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
    {
        "case_name": "hcp_Ti6V2",
        "structure_file": DATA_PATH / "ti64_hcp_phonon_VV_qpath.castep",
        "qpoints_file": DATA_PATH / "ti64_hcp_phonon_VV_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
    {
        "case_name": "hcp_Ti7V",
        "structure_file": DATA_PATH / "Ti7V_hcp_phonon_qpath.castep",
        "qpoints_file": DATA_PATH / "Ti7V_hcp_phonon_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
    {
        "case_name": "bcc_Ti6Al2",
        "structure_file": DATA_PATH / "ti64_bcc_phonon_AlAl_qpath.castep",
        "qpoints_file": DATA_PATH / "ti64_bcc_phonon_AlAl_qpoints.castep",
        "kpath": BCC_KPATH,
        "labels": BCC_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
    {
        "case_name": "bcc_Ti6V2",
        "structure_file": DATA_PATH / "ti64_bcc_phonon_VV_qpath.castep",
        "qpoints_file": DATA_PATH / "ti64_bcc_phonon_VV_qpoints.castep",
        "kpath": BCC_KPATH,
        "labels": BCC_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
    {
        "case_name": "hex_Ti10Al2",
        "structure_file": DATA_PATH / "ti64_hex_phonon_AlAl_qpath.castep",
        "qpoints_file": DATA_PATH / "ti64_hex_phonon_AlAl_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
    {
        "case_name": "hex_Ti10V2",
        "structure_file": DATA_PATH / "ti64_hex_phonon_VV_qpath.castep",
        "qpoints_file": DATA_PATH / "ti64_hex_phonon_VV_qpoints.castep",
        "kpath": HEX_KPATH,
        "labels": HEX_LABELS,
        "grid": GRID_222,
        "disp_phonons": 0.02,
        "disp_pdos": 0.02,
    },
]


MODEL_NAMES = get_model_names(current_models)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_phonon_suite(model_name: str) -> None:
    """
    Run the full Ti64 phonon suite for one model and write artifacts.

    Parameters
    ----------
    model_name
        Name of the ML-PEG model to evaluate.
    """
    patch_ase_castep_reader_dummy_energy(dummy_energy=0.0)

    model = load_models(model_name)[model_name]
    calc = model.get_calculator()

    for spec in CASES:
        case_name = spec["case_name"]
        do_tp = case_name in TP_ON

        run_case(**spec, do_tp=do_tp, calc=calc, model_name=model_name)

        out_dir = OUT_PATH / model_name
        assert (out_dir / f"{case_name}.npz").exists()
        assert (out_dir / f"{case_name}.json").exists()
