"""Run phonon dispersion calculations for the ALEX benchmark."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import pickle
import traceback
from typing import Any

from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.io import write
from ase.optimize import FIRE
from joblib import Parallel, delayed
import numpy as np
from phonopy import load as load_phonopy
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
import pytest
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import yaml

from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import (
    BENCHMARK_DATA_DIR,
    download_alex_parallel,
    get_fc2_and_freqs,
    init_phonopy_from_ref,
    phonopy2aseatoms,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names, load_models

# Keep pytest collection and worker imports lightweight. Reference generation
# does not need model objects, so load model wrappers only inside ``test_phonons``.
MODEL_NAMES = get_model_names(current_models)

OUT_PATH = Path(__file__).parent / "outputs"
DFT_REF_PATH = OUT_PATH / "DFT"

FMAX = 0.005
Q_MESH = 6
Q_MESH_THERMAL = 20
TEMPERATURES = [0, 75, 150, 300, 600]
# Tune to GPU memory: reduce for large models (e.g. UMA-M) or increase for >40 GB GPUs.
N_JOBS = 5
N_PHONONS = 9958
QPATH_METADATA_SUFFIX = "_qpath_metadata.pkl"
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _append_model_log(
    log_path: Path,
    mp_id: str,
    success: bool,
    log_text: str,
) -> None:
    """
    Append one material calculation log section to a model log file.

    Parameters
    ----------
    log_path
        Path to the per-model log file.
    mp_id
        Materials Project identifier for the logged calculation.
    success
        Whether the material calculation completed successfully.
    log_text
        Captured stdout and stderr from the worker process.
    """
    status = "OK" if success else "FAILED"
    with open(log_path, "a") as f:
        f.write(f"\n===== {mp_id} {status} =====\n")
        if log_text:
            f.write(log_text)
            if not log_text.endswith("\n"):
                f.write("\n")


# Crete once per test session + reuse for each test
@pytest.fixture(scope="session")
def alex_phonon_inputs() -> tuple[Path, list[str]]:
    """
    Download ALEX phonon inputs and return the local YAML directory and MP IDs.

    Returns
    -------
    tuple[Path, list[str]]
        Directory containing ``mp-*.yaml`` files and the benchmark MP-ID list.
    """
    download_alex_parallel(sample_every=1)
    yaml_dir = Path(BENCHMARK_DATA_DIR) / "alex_phonons" / "alex_phonon_data"
    ids_file = yaml_dir.parent / "mp_ids_subsampled.txt"
    with open(ids_file) as f:
        mp_ids = [line.strip() for line in f if line.strip()][:N_PHONONS]
    return yaml_dir, mp_ids


def _calc_ref_mp_id(mp_id: str, yaml_dir: Path, out_dir: Path) -> None:
    """
    Generate DFT reference band, DOS, thermal, and q-path metadata for one system.

    Parameters
    ----------
    mp_id
        Materials Project identifier.
    yaml_dir
        Directory containing downloaded ALEX ``phonopy.yaml`` data.
    out_dir
        Directory where serialized DFT reference outputs are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    band_path = out_dir / f"{mp_id}_band_structure.npz"
    dos_path = out_dir / f"{mp_id}_dos.npz"
    thermal_path = out_dir / f"{mp_id}_thermal_properties.json"
    qpath_metadata_path = out_dir / f"{mp_id}{QPATH_METADATA_SUFFIX}"
    struct_path = out_dir / f"{mp_id}.xyz"

    yaml_path = yaml_dir / f"{mp_id}.yaml"

    with open(yaml_path) as f:
        raw = yaml.load(f, Loader=YAML_LOADER)
    thermal = {
        "temperatures": raw["temperatures"],
        "free_energy": raw["free_e"],
        "entropy": raw["entropy"],
        "heat_capacity": raw["heat_capacity"],
    }

    phonons = load_phonopy(str(yaml_path))

    qpoints, labels, connections = get_band_qpoints_by_seekpath(
        phonons.primitive,
        npoints=101,
        is_const_interval=True,
    )

    phonons.run_band_structure(
        paths=qpoints,
        labels=labels,
        path_connections=connections,
    )

    phonons.auto_total_dos()

    band_structure = phonons.get_band_structure_dict()
    band_structure["labels"] = labels
    band_structure["path_connections"] = connections
    with open(band_path, "wb") as f:
        pickle.dump(band_structure, f)
    with open(dos_path, "wb") as f:
        pickle.dump(phonons.get_total_dos_dict(), f)
    with open(thermal_path, "w") as f:
        json.dump(thermal, f, indent=4)
    with open(qpath_metadata_path, "wb") as f:
        pickle.dump(
            {"qpoints": qpoints, "labels": labels, "connections": connections},
            f,
        )
    write(struct_path, phonopy2aseatoms(phonons))


def _ref_complete(mp_id: str, out_dir: Path) -> bool:
    """
    Return True when all DFT reference outputs for ``mp_id`` already exist.

    Parameters
    ----------
    mp_id
        Materials Project identifier.
    out_dir
        Directory containing serialised DFT reference outputs.

    Returns
    -------
    bool
        Whether all expected output files are present.
    """
    return all(
        (out_dir / name).exists()
        for name in (
            f"{mp_id}_band_structure.npz",
            f"{mp_id}_dos.npz",
            f"{mp_id}_thermal_properties.json",
            f"{mp_id}{QPATH_METADATA_SUFFIX}",
            f"{mp_id}.xyz",
        )
    )


@pytest.mark.slow
def test_phonons_ref(alex_phonon_inputs: tuple[Path, list[str]]) -> None:
    """
    Generate DFT reference outputs from ALEX pre-computed force constants.

    Parameters
    ----------
    alex_phonon_inputs
        Downloaded ALEX YAML directory and benchmark MP-ID list.
    """
    yaml_dir, mp_ids = alex_phonon_inputs

    pending = [mp_id for mp_id in mp_ids if not _ref_complete(mp_id, DFT_REF_PATH)]
    n_done = len(mp_ids) - len(pending)
    if not pending:
        print(f"DFT reference complete for all {len(mp_ids)} systems, skipping.")
        return
    print(
        f"DFT reference complete for {n_done}/{len(mp_ids)} systems; "
        f"computing remaining {len(pending)}."
    )

    def handle_mp_id(mp_id: str) -> None:
        """
        Generate DFT reference outputs for one MP ID.

        Parameters
        ----------
        mp_id
            Materials Project identifier to process.
        """
        try:
            _calc_ref_mp_id(mp_id, yaml_dir, DFT_REF_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping reference {mp_id}: {exc}")

    # Prevent each worker spawning N_cores BLAS threads, causing oversubscription.
    with threadpool_limits(limits=1, user_api="blas"):
        Parallel(n_jobs=N_JOBS, batch_size=1)(
            delayed(handle_mp_id)(mp_id)
            for mp_id in tqdm(pending, desc="DFT reference")
        )


def _load_ref_qpath(mp_id: str, ref_dir: Path) -> tuple[Any, Any, Any]:
    """
    Load reference q-path metadata for a system.

    Parameters
    ----------
    mp_id
        Materials Project identifier.
    ref_dir
        Directory containing serialized DFT q-point metadata.

    Returns
    -------
    tuple[Any, Any, Any]
        Q-points, labels, and path connections for ``phonons.run_band_structure``.
    """
    qpath_metadata_path = ref_dir / f"{mp_id}{QPATH_METADATA_SUFFIX}"
    with open(qpath_metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return metadata["qpoints"], metadata["labels"], metadata["connections"]


def _calc_mp_id(
    mp_id: str,
    calc: Calculator,
    yaml_dir: Path,
    ref_dir: Path,
    out_dir: Path,
) -> None:
    """
    Calculate model phonon outputs for one ALEX material.

    Parameters
    ----------
    mp_id
        Materials Project identifier.
    calc
        ASE calculator used for relaxation and displacement forces.
    yaml_dir
        Directory containing downloaded ALEX ``phonopy.yaml`` data.
    ref_dir
        Directory containing DFT reference q-path metadata.
    out_dir
        Directory where model prediction outputs are written.
    """
    band_path = out_dir / f"{mp_id}_band_structure.npz"
    dos_path = out_dir / f"{mp_id}_dos.npz"
    thermal_path = out_dir / f"{mp_id}_thermal_properties.json"
    struct_path = out_dir / f"{mp_id}.xyz"

    if (
        band_path.exists()
        and dos_path.exists()
        and thermal_path.exists()
        and struct_path.exists()
    ):
        print(f"Skipping {mp_id}: already completed.")
        return

    yaml_path = yaml_dir / f"{mp_id}.yaml"
    phonons = load_phonopy(str(yaml_path))
    displacement_dataset = phonons.dataset

    atoms = phonopy2aseatoms(phonons)
    atoms_sym = atoms.copy()
    atoms_sym.info.setdefault("charge", 0)
    atoms_sym.info.setdefault("spin", 1)
    atoms_sym.calc = calc
    atoms_sym.set_constraint(FixSymmetry(atoms_sym))
    FIRE(atoms_sym).run(fmax=FMAX, steps=1000)

    if "primitive_matrix" in atoms.info:
        primitive_matrix = atoms.info["primitive_matrix"]
    else:
        unitcell = phonons.unitcell
        primitive_matrix = np.linalg.inv(np.array(unitcell.cell)) @ np.array(
            phonons.primitive.cell
        )

    phonons = init_phonopy_from_ref(
        atoms=atoms_sym,
        fc2_supercell=atoms.info["fc2_supercell"],
        primitive_matrix=primitive_matrix,
        displacement_dataset=displacement_dataset,
        symprec=1e-5,
    )
    phonons, _, _ = get_fc2_and_freqs(
        phonons=phonons,
        calculator=calc,
        q_mesh=np.array([Q_MESH] * 3),
        symmetrize_fc2=True,
    )

    qpoints, labels, connections = _load_ref_qpath(mp_id, ref_dir)
    phonons.run_band_structure(
        paths=qpoints,
        labels=labels,
        path_connections=connections,
    )

    with open(band_path, "wb") as f:
        pickle.dump(phonons.get_band_structure_dict(), f)

    phonons.auto_total_dos()
    with open(dos_path, "wb") as f:
        pickle.dump(phonons.get_total_dos_dict(), f)

    phonons.run_mesh([Q_MESH_THERMAL] * 3)
    phonons.run_thermal_properties(temperatures=TEMPERATURES)
    thermal = phonons.get_thermal_properties_dict()
    thermal_safe = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in thermal.items()
    }
    with open(thermal_path, "w") as f:
        json.dump(thermal_safe, f, indent=4)
    atoms_sym.calc = None
    write(struct_path, atoms_sym)


def _model_complete(mp_id: str, out_dir: Path) -> bool:
    """
    Return True when all model phonon outputs for ``mp_id`` already exist.

    Parameters
    ----------
    mp_id
        Materials Project identifier.
    out_dir
        Directory containing model phonon outputs.

    Returns
    -------
    bool
        Whether all expected output files are present.
    """
    return all(
        (out_dir / name).exists()
        for name in (
            f"{mp_id}_band_structure.npz",
            f"{mp_id}_dos.npz",
            f"{mp_id}_thermal_properties.json",
            f"{mp_id}.xyz",
        )
    )


def _calc_model_mp_id(
    mp_id: str,
    model: Any,
    yaml_dir: Path,
    ref_dir: Path,
    out_dir: Path,
) -> tuple[str, bool, str]:
    """
    Load a model calculator in a worker process and calculate one phonon system.

    Parameters
    ----------
    mp_id
        Materials Project identifier.
    model
        ML-PEG model wrapper used to construct an ASE calculator.
    yaml_dir
        Directory containing downloaded ALEX ``phonopy.yaml`` data.
    ref_dir
        Directory containing DFT reference q-path metadata.
    out_dir
        Directory where model prediction outputs are written.

    Returns
    -------
    tuple[str, bool, str]
        MP ID, success flag, and captured stdout/stderr text.
    """
    log = StringIO()
    success = True
    with redirect_stdout(log), redirect_stderr(log):
        try:
            calc = model.get_calculator()
            _calc_mp_id(mp_id, calc, yaml_dir, ref_dir, out_dir)
        except Exception:  # noqa: BLE001
            success = False
            traceback.print_exc()
    return mp_id, success, log.getvalue()


@pytest.mark.very_slow
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_phonons(
    model_name: str,
    alex_phonon_inputs: tuple[Path, list[str]],
) -> None:
    """
    Run phonon dispersion calculations for one model over all ALEX MP IDs.

    Parameters
    ----------
    model_name
        Name of the ML-PEG model to evaluate.
    alex_phonon_inputs
        Downloaded ALEX YAML directory and benchmark MP-ID list.
    """
    if not DFT_REF_PATH.exists():
        pytest.skip("DFT reference not found — run test_phonons_ref first.")

    model = load_models(model_name)[model_name]
    yaml_dir, mp_ids = alex_phonon_inputs

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"0_{model_name}.log"
    with open(log_path, "w") as f:
        f.write(f"Phonon calculation log for {model_name}\n")

    pending = [mp_id for mp_id in mp_ids if not _model_complete(mp_id, out_dir)]
    n_done = len(mp_ids) - len(pending)
    if not pending:
        print(f"All {len(mp_ids)} phonon calculations complete, skipping.")
        return
    print(
        f"Phonon calculations complete for {n_done}/{len(mp_ids)} systems; "
        f"computing remaining {len(pending)}."
    )

    results = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(
        delayed(_calc_model_mp_id)(mp_id, model, yaml_dir, DFT_REF_PATH, out_dir)
        for mp_id in pending
    )

    # Large structures can (infrequently) OOM when multiple workers share the GPU.
    # Catch them and retry sequentially so only one system is in GPU memory at a time.
    oom_mp_ids = []
    failed_mp_ids = []
    with tqdm(total=len(pending), desc=f"{model_name} phonons", unit="phonon") as pbar:
        for mp_id, success, log_text in results:
            _append_model_log(log_path, mp_id, success, log_text)
            if not success:
                if "cuda out of memory" in log_text.lower():
                    oom_mp_ids.append(mp_id)
                else:
                    failed_mp_ids.append(mp_id)
            pbar.update()

    if oom_mp_ids:
        tqdm.write(f"{model_name}: retrying {len(oom_mp_ids)} OOM systems sequentially")
        for mp_id in tqdm(oom_mp_ids, desc=f"{model_name} OOM retry", unit="phonon"):
            mp_id, success, log_text = _calc_model_mp_id(
                mp_id, model, yaml_dir, DFT_REF_PATH, out_dir
            )
            _append_model_log(log_path, mp_id, success, log_text)
            if not success:
                failed_mp_ids.append(mp_id)

    if failed_mp_ids:
        tqdm.write(
            f"{model_name}: skipped {len(failed_mp_ids)} phonons; see {log_path}"
        )
