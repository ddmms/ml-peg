"""Track completed benchmark calculations so they can be skipped on re-runs."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any

MARKER_FILENAME = ".completed.json"

# Benchmark data files recorded since the current test started
_data_files: set[str] = set()

# Content hashes of local files (e.g. model checkpoints), keyed by path, size
# and mtime so each file is only re-hashed after it changes
_file_hashes: dict[tuple[str, int, int], str] = {}


def record_data_file(path: Path | str) -> None:
    """
    Record a benchmark data file used by the current test.

    Called by the download utilities so that completion markers can store
    which cached data files a calculation depends on.

    Parameters
    ----------
    path
        Path to the locally cached data file.
    """
    _data_files.add(str(path))


def clear_data_files() -> None:
    """Reset the record of benchmark data files used by the current test."""
    _data_files.clear()


def used_data_files() -> list[str]:
    """
    Get benchmark data files recorded since the last reset.

    Returns
    -------
    list[str]
        Sorted paths of recorded data files.
    """
    return sorted(_data_files)


def _model_config(model_name: str) -> dict[str, Any]:
    """
    Get a model's configuration from the models file.

    Parameters
    ----------
    model_name
        Name of the model to look up.

    Returns
    -------
    dict[str, Any]
        Model configuration, or an empty dict if not defined (e.g. "mock").
    """
    from ml_peg import models
    from ml_peg.models.get_models import _load_models_yaml

    config = _load_models_yaml(models.models_file).get(model_name)
    return config if isinstance(config, dict) else {}


def _local_files(config: Any) -> list[Path]:
    """
    Find existing local files referenced in a model configuration.

    Parameters
    ----------
    config
        Model configuration, or a nested value within it.

    Returns
    -------
    list[Path]
        Paths of existing local files, such as model checkpoints.
    """
    if isinstance(config, dict):
        return [file for value in config.values() for file in _local_files(value)]
    if isinstance(config, (list, tuple)):
        return [file for value in config for file in _local_files(value)]
    if isinstance(config, str):
        try:
            path = Path(config)
            if path.is_file():
                return [path]
        except (OSError, ValueError):
            pass
    return []


def _file_sha256(path: Path) -> str:
    """
    Hash a file's contents, cached per process.

    Parameters
    ----------
    path
        Path of the file to hash.

    Returns
    -------
    str
        Hex digest of the file's contents.
    """
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _file_hashes:
        sha = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                sha.update(chunk)
        _file_hashes[key] = sha.hexdigest()
    return _file_hashes[key]


def calc_fingerprint(
    calc_dir: Path, model_name: str, config: dict[str, Any] | None = None
) -> str:
    """
    Fingerprint a calculation's inputs for one model.

    Hashes the Python source files in the benchmark directory, the model's
    configuration, and the contents of any local files the configuration
    references, such as model checkpoints.

    Parameters
    ----------
    calc_dir
        Directory containing the benchmark's calculation script(s).
    model_name
        Name of the model the calculation runs with.
    config
        Model configuration. Default is the model's entry in the models file.

    Returns
    -------
    str
        Hex digest identifying the calculation's inputs.
    """
    config = _model_config(model_name) if config is None else config

    sha = hashlib.sha256()
    for source in sorted(Path(calc_dir).glob("*.py")):
        sha.update(source.name.encode())
        sha.update(source.read_bytes())
    sha.update(json.dumps(config, sort_keys=True, default=str).encode())
    for path in _local_files(config):
        sha.update(f"{path}:{_file_sha256(path)}".encode())
    return sha.hexdigest()


def analysis_fingerprint(
    analysis_dir: Path,
    model_names: Iterable[str],
    calc_path: Path,
    configs: dict[str, dict[str, Any]] | None = None,
) -> str:
    """
    Fingerprint an analysis benchmark's inputs across all models.

    Hashes the Python and YAML sources in the analysis directory, each model's
    configuration, and the raw bytes of each model's calculation completion
    marker. Re-running a calculation rewrites its marker, so this invalidates
    the analysis transitively without hashing the calculation outputs
    themselves. Local files referenced by model configurations are not hashed
    either, as the calculation markers already reflect them.

    Parameters
    ----------
    analysis_dir
        Directory containing the benchmark's analysis script(s).
    model_names
        Names of the models the analysis aggregates.
    calc_path
        Benchmark calculation outputs directory, containing per-model
        completion markers.
    configs
        Model configurations by name. Default is each model's entry in the
        models file.

    Returns
    -------
    str
        Hex digest identifying the analysis' inputs.
    """
    sha = hashlib.sha256()
    for source in sorted(
        [*Path(analysis_dir).glob("*.py"), *Path(analysis_dir).glob("*.yml")]
    ):
        sha.update(source.name.encode())
        sha.update(source.read_bytes())
    for name in sorted(model_names):
        config = _model_config(name) if configs is None else configs.get(name, {})
        sha.update(name.encode())
        sha.update(json.dumps(config, sort_keys=True, default=str).encode())
        marker = Path(calc_path) / name / MARKER_FILENAME
        if marker.is_file():
            sha.update(marker.read_bytes())
        else:
            sha.update(f"absent:{name}".encode())
    return sha.hexdigest()


def _read_marker(marker: Path) -> dict[str, Any]:
    """
    Read a completion marker file.

    Parameters
    ----------
    marker
        Path to the marker file.

    Returns
    -------
    dict[str, Any]
        Marker contents, or an empty dict if missing or unreadable.
    """
    try:
        content = json.loads(marker.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return content if isinstance(content, dict) else {}


def is_complete(
    out_path: Path, model_name: str, test_name: str, fingerprint: str
) -> bool:
    """
    Check whether a calculation previously completed with identical inputs.

    Data files are considered unchanged while their cached copies exist,
    matching the criterion the download utilities use to reuse the cache.

    Parameters
    ----------
    out_path
        Benchmark outputs directory.
    model_name
        Name of the model the calculation runs with.
    test_name
        Name of the test running the calculation.
    fingerprint
        Current fingerprint of the calculation's inputs.

    Returns
    -------
    bool
        Whether the calculation previously completed with identical inputs.
    """
    entry = _read_marker(out_path / model_name / MARKER_FILENAME).get(test_name)
    if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
        return False
    return all(Path(path).exists() for path in entry.get("data_files", ()))


def mark_complete(
    out_path: Path,
    model_name: str,
    test_name: str,
    fingerprint: str,
    data_files: list[str],
) -> None:
    """
    Mark a calculation as completed.

    Parameters
    ----------
    out_path
        Benchmark outputs directory.
    model_name
        Name of the model the calculation ran with.
    test_name
        Name of the test that ran the calculation.
    fingerprint
        Fingerprint of the calculation's inputs.
    data_files
        Paths of cached data files the calculation used.
    """
    marker = out_path / model_name / MARKER_FILENAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    content = _read_marker(marker)
    content[test_name] = {"fingerprint": fingerprint, "data_files": list(data_files)}
    marker.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf8")


def unmark_complete(out_path: Path, model_name: str, test_name: str) -> None:
    """
    Remove a test's completion marker entry, if present.

    Called before a test re-runs, so that a failed or interrupted run cannot
    leave outputs masked as complete by an earlier success.

    Parameters
    ----------
    out_path
        Benchmark outputs directory.
    model_name
        Name of the model the calculation ran with.
    test_name
        Name of the test to remove the marker entry for.
    """
    marker = out_path / model_name / MARKER_FILENAME
    content = _read_marker(marker)
    if test_name not in content:
        return
    del content[test_name]
    marker.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf8")
