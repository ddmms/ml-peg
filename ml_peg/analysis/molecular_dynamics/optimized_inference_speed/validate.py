"""Validate optimised inference runs against the contract and reference models."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
import numpy as np
from pydantic import ValidationError

from ml_peg.app import APP_ROOT
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.contract import (
    RESULT_FILE,
    VALIDATION_FILE,
    InferenceResult,
)

ELEMENT_COVERAGE = APP_ROOT / "data" / "element_coverage.json"

# Static accuracy tolerances: (energy in meV/atom, force RMSE in meV/Å)
TOLERANCES = {"fp64": (0.1, 1.0), "fp32": (1.0, 10.0), "mixed": (1.0, 10.0)}
POSITION_TOLERANCE = 1e-4  # Å


@dataclasses.dataclass
class Check:
    """Outcome of one validation check."""

    name: str
    ok: bool
    reason: str = ""


def check_schema(run_dir: Path) -> tuple[InferenceResult | None, Check]:
    """
    Parse ``inference.json`` with the contract schema.

    Parameters
    ----------
    run_dir
        Run output directory.

    Returns
    -------
    tuple[InferenceResult | None, Check]
        Parsed result, if valid, and the check.
    """
    path = run_dir / RESULT_FILE
    try:
        result = InferenceResult.model_validate_json(path.read_text(encoding="utf8"))
    except FileNotFoundError:
        return None, Check("schema", False, f"{RESULT_FILE} missing")
    except ValidationError as err:
        errors = "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()
        )
        return None, Check("schema", False, f"schema: {errors}")
    return result, Check("schema", True)


def check_identity(
    result: InferenceResult, record: dict[str, Any], container: dict[str, Any]
) -> Check:
    """
    Check the model, weights and run match what was launched.

    Parameters
    ----------
    result
        Container result.
    record
        ML-PEG run record, including the container's labels.
    container
        Container metadata from the benchmark's ``models.yml``.

    Returns
    -------
    Check
        Identity check.
    """
    labels = record["labels"]
    problems = []
    hashes = {
        "inference.json": result.model.weights_sha256,
        "label": labels.get("mlpeg.weights_sha256"),
        "models.yml": container.get("weights_sha256"),
    }
    if len(set(hashes.values())) != 1:
        problems.append(
            "weights_sha256 mismatch ("
            + ", ".join(f"{k}={str(v)[:12]}" for k, v in hashes.items())
            + ")"
        )
    names = {result.model.name, labels.get("mlpeg.model"), container.get("mlpeg_model")}
    if len(names) != 1:
        problems.append(f"model name mismatch {sorted(map(str, names))}")
    if result.model.precision != record["precision"]:
        problems.append(f"precision {result.model.precision} != {record['precision']}")
    if result.run_id != record["run_id"]:
        problems.append(f"run_id {result.run_id} != {record['run_id']}")
    if result.system.n_atoms != record["n_atoms"]:
        problems.append(f"n_atoms {result.system.n_atoms} != {record['n_atoms']}")
    return Check("identity", not problems, "; ".join(problems))


def get_supported_elements(container: dict[str, Any]) -> set[str] | None:
    """
    Get the elements a container's model supports.

    Parameters
    ----------
    container
        Container metadata from the benchmark's ``models.yml``.

    Returns
    -------
    set[str] | None
        Supported elements, or None if any element is supported.
    """
    if container.get("elements_from_dataset"):
        coverage = json.loads(ELEMENT_COVERAGE.read_text(encoding="utf8"))
        return set(
            coverage["datasets"][container["elements_from_dataset"]]["supported"]
        )
    elements = container.get("elements")
    return None if elements is None else set(elements)


def check_elements(
    result: InferenceResult, elements: list[str], supported: set[str] | None
) -> Check:
    """
    Check an ``unsupported_elements`` status agrees with the model's elements.

    Parameters
    ----------
    result
        Container result.
    elements
        Elements in the system.
    supported
        Elements supported by the model, or None for any element.

    Returns
    -------
    Check
        Element support check.
    """
    missing = sorted(set(elements) - supported) if supported is not None else []
    if result.status == "unsupported_elements" and not missing:
        return Check(
            "elements",
            False,
            f"reported unsupported_elements, but {elements} supported",
        )
    if result.status == "ok" and missing:
        return Check("elements", False, f"ran with unsupported elements {missing}")
    return Check("elements", True)


def check_drift(result: InferenceResult, n_atoms: int, max_mev: float = 1.0) -> Check:
    """
    Check NVE total energy drift over the measured phase.

    Parameters
    ----------
    result
        Container result with status ``ok``.
    n_atoms
        Number of atoms.
    max_mev
        Maximum drift in meV/atom.

    Returns
    -------
    Check
        Drift check.
    """
    drift = (
        abs(result.energy.total_end_eV - result.energy.total_start_eV) / n_atoms * 1000
    )
    if drift > max_mev:
        return Check("drift", False, f"NVE drift {drift:.3g} > {max_mev} meV/atom")
    return Check("drift", True)


def check_timing(
    result: InferenceResult, wall_time_s: float, min_steps: int = 20
) -> Check:
    """
    Check timestamps are monotonic, within the wall time, with enough steps.

    Parameters
    ----------
    result
        Container result with status ``ok``.
    wall_time_s
        Wall time measured by ML-PEG, from launch to exit.
    min_steps
        Minimum number of measured steps.

    Returns
    -------
    Check
        Timing check.
    """
    stamps = result.timestamps_unix
    problems = []
    if not (stamps.start <= stamps.setup_end <= stamps.warmup_end < stamps.measure_end):
        problems.append("timestamps not monotonic")
    if stamps.measure_end - stamps.start > wall_time_s:
        problems.append(
            f"measure_end - start = {stamps.measure_end - stamps.start:.3f} s exceeds "
            f"wall time {wall_time_s:.3f} s"
        )
    if result.steps.measured < min_steps:
        problems.append(f"{result.steps.measured} measured steps < {min_steps}")
    return Check("timing", not problems, "; ".join(problems))


def check_static(
    validation_path: Path,
    structure: Atoms,
    reference: Calculator,
    precision: str,
) -> Check:
    """
    Compare energy and forces at the input positions against the reference model.

    Parameters
    ----------
    validation_path
        Container's ``validation.extxyz``.
    structure
        Staged input structure.
    reference
        Reference calculator, in float64.
    precision
        Precision mode of the run, which sets the tolerances.

    Returns
    -------
    Check
        Static accuracy check.
    """
    if not validation_path.is_file():
        return Check("static", False, f"{VALIDATION_FILE} missing")
    try:
        frame = read(validation_path, format="extxyz")
        energy = frame.get_potential_energy()
        forces = frame.get_forces()
    except Exception as err:  # noqa: BLE001 - any unreadable file is invalid
        return Check("static", False, f"{VALIDATION_FILE} unreadable: {err}")
    if len(frame) != len(structure) or not np.allclose(
        frame.positions, structure.positions, atol=POSITION_TOLERANCE
    ):
        return Check("static", False, "positions differ from input structure")

    atoms = Atoms(
        numbers=structure.numbers,
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )
    atoms.calc = reference
    ref_energy = atoms.get_potential_energy()
    ref_forces = atoms.get_forces()

    energy_err = abs(energy - ref_energy) / len(atoms) * 1000
    force_rmse = float(np.sqrt(np.mean((forces - ref_forces) ** 2))) * 1000
    energy_tol, force_tol = TOLERANCES[precision]
    problems = []
    if energy_err > energy_tol:
        problems.append(f"energy error {energy_err:.3g} > {energy_tol} meV/atom")
    if force_rmse > force_tol:
        problems.append(f"force RMSE {force_rmse:.3g} > {force_tol} meV/Å")
    return Check("static", not problems, "; ".join(problems))


@functools.cache
def get_reference_calculator(mlpeg_model: str) -> Calculator:
    """
    Get ML-PEG's float64 reference calculator for a model.

    Parameters
    ----------
    mlpeg_model
        ML-PEG model name; ``mock`` is the zero calculator.

    Returns
    -------
    Calculator
        Reference calculator.
    """
    from ml_peg.models.get_models import load_models

    if mlpeg_model == "mock":
        model = load_models(run_mock=True, mock_only=True)["mock"]
    else:
        model = load_models([mlpeg_model], run_mock=False)[mlpeg_model]
    return model.get_calculator(precision="high")


def validate_run(
    run_dir: Path,
    record: dict[str, Any],
    spec: dict[str, Any],
    container: dict[str, Any],
    reference_getter: Callable[[str], Calculator] = get_reference_calculator,
) -> tuple[bool | None, list[Check], InferenceResult | None]:
    """
    Validate one run.

    Runs that ML-PEG recorded as ``error``, ``timeout`` or ``skipped`` are not
    validated. Otherwise every applicable check must pass for the run to count.

    Parameters
    ----------
    run_dir
        Run output directory.
    record
        ML-PEG run record.
    spec
        Campaign spec.
    container
        Container metadata from the benchmark's ``models.yml``.
    reference_getter
        Function returning the reference calculator for an ML-PEG model name.

    Returns
    -------
    tuple[bool | None, list[Check], InferenceResult | None]
        Whether the run is valid (None if not validated), the checks, and the
        parsed result.
    """
    if record["status"] in ("error", "timeout") or record.get("exit_code") is None:
        return None, [], None

    result, schema = check_schema(run_dir)
    checks = [schema]
    if result is None:
        return False, checks, None

    system = next(s for s in spec["systems"] if s["material"] == record["material"])
    size = next(s for s in system["sizes"] if s["n_atoms"] == record["n_atoms"])
    checks.append(check_identity(result, record, container))
    checks.append(
        check_elements(result, system["elements"], get_supported_elements(container))
    )
    if result.status == "ok":
        checks.append(
            check_drift(
                result,
                record["n_atoms"],
                spec["validation"].get("nve_drift_max_meV_per_atom", 1.0),
            )
        )
        checks.append(
            check_timing(
                result,
                record["wall_time_s"],
                spec["md"].get("min_measured_steps", 20),
            )
        )
        if size.get("validate"):
            structure = read(Path(run_dir).parents[3] / "in" / size["file"])
            checks.append(
                check_static(
                    run_dir / VALIDATION_FILE,
                    structure,
                    reference_getter(record["mlpeg_model"]),
                    record["precision"],
                )
            )
    return all(check.ok for check in checks), checks, result
