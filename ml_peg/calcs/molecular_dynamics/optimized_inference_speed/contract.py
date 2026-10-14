"""Contract v1.0 between ML-PEG and optimised inference containers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

CONTRACT_VERSION = "1.0"
SCHEMA_VERSION = "1.0"

BENCHMARK_DIR = Path(__file__).parent
SPEC_TEMPLATE = BENCHMARK_DIR / "spec_template.json"
CONTAINER_MODELS_FILE = BENCHMARK_DIR / "models.yml"
CONTRIB_ROOT = BENCHMARK_DIR.parents[3] / "contrib" / "optimized_inference_speed"

# Paths inside the container
CONTAINER_IN = "/mlpeg/in"
CONTAINER_OUT = "/mlpeg/out"

RESULT_FILE = "inference.json"
VALIDATION_FILE = "validation.extxyz"
RECORD_FILE = "mlpeg_record.json"

PRECISIONS = ("fp64", "fp32", "mixed")
STATUSES = ("ok", "oom", "unsupported_elements", "skipped", "error")
REQUIRED_LABELS = (
    "mlpeg.model",
    "mlpeg.precisions",
    "mlpeg.weights_sha256",
    "mlpeg.contract_version",
)

RUN_ID_PATTERN = re.compile(r"^(?P<material>.+)-n(?P<n_atoms>\d+)-r(?P<repeat>\d+)$")


def format_run_id(material: str, n_atoms: int, repeat: int) -> str:
    """
    Format a run ID as ``<material>-n<n_atoms>-r<repeat>``.

    Parameters
    ----------
    material
        Material identifier.
    n_atoms
        Number of atoms in the structure.
    repeat
        Repeat index, starting at 1.

    Returns
    -------
    str
        Run ID.
    """
    return f"{material}-n{n_atoms}-r{repeat}"


class ModelInfo(BaseModel):
    """Model identity reported by the container."""

    name: str
    weights_sha256: str
    precision: Literal["fp64", "fp32", "mixed"]


class ImplementationInfo(BaseModel):
    """Implementation reported by the container."""

    engine: str
    engine_version: str | None = None
    repo: str | None = None
    commit: str | None = None
    settings: dict[str, Any] = {}


class SystemInfo(BaseModel):
    """System that was simulated."""

    material: str
    n_atoms: int


class Timestamps(BaseModel):
    """Unix timestamps of each phase; later phases may be missing on failure."""

    start: float
    setup_end: float | None = None
    warmup_end: float | None = None
    measure_end: float | None = None


class Steps(BaseModel):
    """Number of MD steps taken in each phase."""

    warmup: int
    measured: int


class Energies(BaseModel):
    """Total energies at the start and end of the measured phase."""

    total_start_eV: float  # noqa: N815
    total_end_eV: float  # noqa: N815


class InferenceResult(BaseModel):
    """Schema of ``inference.json`` written by the container."""

    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["1.0"]
    run_id: str
    status: Literal["ok", "oom", "unsupported_elements", "skipped", "error"]
    error: str | None = None
    model: ModelInfo
    implementation: ImplementationInfo
    system: SystemInfo
    timestamps_unix: Timestamps
    steps: Steps | None = None
    energy: Energies | None = None
    self_reported: dict[str, Any] = {}

    @model_validator(mode="after")
    def check_complete_if_ok(self) -> InferenceResult:
        """
        Require all timings, steps and energies for successful runs.

        Returns
        -------
        InferenceResult
            Validated result.
        """
        if self.status != "ok":
            return self
        stamps = self.timestamps_unix
        missing = [
            name
            for name in ("setup_end", "warmup_end", "measure_end")
            if getattr(stamps, name) is None
        ]
        if self.steps is None:
            missing.append("steps")
        if self.energy is None:
            missing.append("energy")
        if missing:
            raise ValueError(f"status 'ok' requires {', '.join(missing)}")
        return self


def build_invocation(
    sif: Path,
    campaign_in: Path,
    run_dir: Path,
    run_id: str,
    precision: str,
    extra_env: Sequence[str] = (),
) -> list[str]:
    """
    Build the contract's ``apptainer run`` command for a single run.

    Parameters
    ----------
    sif
        Path to the container image.
    campaign_in
        Campaign input directory, bound read-only to ``/mlpeg/in``.
    run_dir
        Run output directory, bound to ``/mlpeg/out``.
    run_id
        Run ID, ``<material>-n<n_atoms>-r<repeat>``.
    precision
        Precision mode.
    extra_env
        Extra ``KEY=VALUE`` environment variables appended to the contract's.

    Returns
    -------
    list[str]
        Command to launch.
    """
    env = ",".join(["CUDA_VISIBLE_DEVICES=0", "OMP_NUM_THREADS=8", *extra_env])
    return [
        "apptainer",
        "run",
        "--nv",
        "--containall",
        "--cleanenv",
        "--net",
        "--network",
        "none",
        "--bind",
        f"{campaign_in}:{CONTAINER_IN}:ro",
        "--bind",
        f"{run_dir}:{CONTAINER_OUT}",
        "--env",
        env,
        str(sif),
        "--spec",
        f"{CONTAINER_IN}/spec.json",
        "--run-id",
        run_id,
        "--precision",
        precision,
        "--out",
        CONTAINER_OUT,
    ]
