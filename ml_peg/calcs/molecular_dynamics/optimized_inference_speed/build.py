"""Build optimised inference containers and check their labels."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.contract import (
    CONTRACT_VERSION,
    CONTRIB_ROOT,
    REQUIRED_LABELS,
)
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.stage import (
    sha256_file,
)

DEFAULT_CACHE = Path(
    os.environ.get(
        "MLPEG_SIF_CACHE",
        Path.home() / ".cache" / "ml_peg" / "optimized_inference_speed",
    )
)


def parse_def_labels(def_path: Path) -> dict[str, str]:
    """
    Parse the ``%labels`` section of an Apptainer definition file.

    Parameters
    ----------
    def_path
        Path to ``container.def``.

    Returns
    -------
    dict[str, str]
        Labels defined in the file.
    """
    labels = {}
    in_labels = False
    for line in def_path.read_text(encoding="utf8").splitlines():
        stripped = line.strip()
        if stripped.startswith("%"):
            in_labels = stripped.split()[0] == "%labels"
            continue
        if in_labels and stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition(" ")
            labels[key] = value.strip()
    return labels


def check_labels(labels: dict[str, str]) -> list[str]:
    """
    Check required contract labels are present and consistent.

    Parameters
    ----------
    labels
        Container labels.

    Returns
    -------
    list[str]
        Problems found. Empty if the labels are valid.
    """
    problems = [f"missing label {key}" for key in REQUIRED_LABELS if key not in labels]
    version = labels.get("mlpeg.contract_version")
    if version is not None and version != CONTRACT_VERSION:
        problems.append(f"contract version {version} != {CONTRACT_VERSION}")
    return problems


def inspect_labels(sif: Path) -> dict[str, str]:
    """
    Read labels from a built image with ``apptainer inspect``.

    Parameters
    ----------
    sif
        Path to the image.

    Returns
    -------
    dict[str, str]
        Image labels.
    """
    result = subprocess.run(
        ["apptainer", "inspect", "--labels", "--json", str(sif)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["data"]["attributes"]["labels"]


def build_container(
    model: str,
    cache_dir: Path = DEFAULT_CACHE,
    contrib_dir: Path = CONTRIB_ROOT,
) -> dict:
    """
    Build ``<cache_dir>/<model>.sif`` and record its checksum and labels.

    Parameters
    ----------
    model
        Container name, a directory under `contrib_dir`.
    cache_dir
        Directory to write the image and build record to.
    contrib_dir
        Directory containing container definitions.

    Returns
    -------
    dict
        Build record, also written to ``<cache_dir>/<model>.build.json``.
    """
    def_path = contrib_dir / model / "container.def"
    if not def_path.is_file():
        raise FileNotFoundError(f"No container definition at {def_path}")
    problems = check_labels(parse_def_labels(def_path))
    if problems:
        raise ValueError(f"{def_path}: {'; '.join(problems)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    sif = cache_dir / f"{model}.sif"
    # Build from the definition's directory so relative %files paths resolve
    subprocess.run(
        ["apptainer", "build", "--force", str(sif), def_path.name],
        cwd=def_path.parent,
        check=True,
    )
    labels = inspect_labels(sif)
    problems = check_labels(labels)
    if problems:
        raise ValueError(f"{sif}: {'; '.join(problems)}")

    record = {"model": model, "sif": str(sif), "sha256": sha256_file(sif)}
    record["labels"] = labels
    (cache_dir / f"{model}.build.json").write_text(json.dumps(record, indent=2))
    return record
