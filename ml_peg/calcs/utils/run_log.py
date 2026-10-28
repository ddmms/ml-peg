"""Record when calculations ran, and whether they completed."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Literal
import warnings

from ml_peg.calcs import CALCS_ROOT

#: Version of the run log format, to allow analysis to reject logs it cannot read.
SCHEMA_VERSION = 1

#: Name of the log file written within each model's output directory.
RUN_LOG_NAME = "run_log.json"

#: Status of a calculation. A log left as "running" indicates the process died
#: without ever reaching `end_run`, such as a scheduler timeout or an OOM kill.
Status = Literal["running", "completed", "failed", "skipped"]

#: Maximum number of distinct warnings recorded per run, and maximum length of each
#: warning message, to keep logs readable when a benchmark warns per structure.
MAX_UNIQUE_WARNINGS = 20
MAX_WARNING_CHARS = 300

#: Maximum length of a recorded error message and traceback.
MAX_ERROR_CHARS = 500
MAX_TRACEBACK_CHARS = 4000

#: Packages whose installed versions affect results for every model.
CORE_PACKAGES = ("ase", "numpy", "torch")


def benchmark_id(path: Path) -> str | None:
    """
    Get the "category/test" identifier for a calculation script or directory.

    Parameters
    ----------
    path
        Path to a ``calc_*.py`` script, or to the benchmark directory containing it.

    Returns
    -------
    str | None
        Identifier of the benchmark, or `None` if `path` is not a benchmark.
    """
    directory = path.parent if path.suffix == ".py" else path
    if not directory.is_relative_to(CALCS_ROOT):
        return None
    parts = directory.relative_to(CALCS_ROOT).parts
    if len(parts) != 2:
        return None
    return "/".join(parts)


def _timestamp() -> str:
    """
    Get the current time as a UTC timestamp.

    Returns
    -------
    str
        Current time, in ISO 8601 format.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_git(*args: str) -> str | None:
    """
    Run a git command in the repository containing ml_peg.

    Parameters
    ----------
    *args
        Arguments passed to git.

    Returns
    -------
    str | None
        Stripped standard output, or `None` if git failed or is unavailable.
    """
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=CALCS_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(name: str) -> str | None:
    """
    Get the installed version of a package without importing it.

    Parameters
    ----------
    name
        Name of the package.

    Returns
    -------
    str | None
        Installed version, or `None` if the package is not installed.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _static_environment() -> dict[str, Any]:
    """
    Describe the code and environment the calculations are running from.

    Cached, as nothing recorded here can change within a single process.

    Returns
    -------
    dict[str, Any]
        Versions, source revision and host identifying the run.
    """
    from ml_peg import __version__

    commit = _run_git("rev-parse", "HEAD")
    # uv.lock pins the resolved environment, but a run may not have been synced from
    # it, so the installed versions below are recorded as well
    lock = CALCS_ROOT.parent.parent / "uv.lock"

    return {
        "ml_peg_version": __version__,
        "git_commit": commit,
        "git_dirty": bool(_run_git("status", "--porcelain")) if commit else None,
        "uv_lock_sha256": _file_digest(lock),
        "python_version": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in CORE_PACKAGES
            if _package_version(name)
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _file_digest(path: Path) -> str | None:
    """
    Get the SHA256 digest of a file.

    Parameters
    ----------
    path
        Path to the file to hash.

    Returns
    -------
    str | None
        Hex digest, or `None` if the file does not exist.
    """
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _device() -> str | None:
    """
    Get the accelerator in use, if torch has been loaded and sees one.

    Returns
    -------
    str | None
        Name of the CUDA device, or `None` if torch is unused or on CPU.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return None
    return None


def model_config(model_name: str) -> dict[str, Any]:
    """
    Describe the configuration a model was loaded from.

    Parameters
    ----------
    model_name
        Name of the model, as defined in the models file.

    Returns
    -------
    dict[str, Any]
        Model source and a digest of its configuration, used by analysis to detect
        results produced by a since-changed model definition.
    """
    from ml_peg.models.get_models import _load_models_yaml, default_models_file

    try:
        config = _load_models_yaml().get(model_name)
    except (OSError, ValueError):
        config = None

    if config is None:
        return {"name": model_name, "config_sha256": None}

    module = str(config.get("module", "")).split(".")[0]
    return {
        "name": model_name,
        "models_file": str(default_models_file()),
        "module": config.get("module"),
        "module_version": _package_version(module) if module else None,
        "config_sha256": sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }


def summarise_warnings(counts: Counter[tuple[str, str]]) -> dict[str, Any]:
    """
    Summarise the warnings emitted by a run.

    Benchmarks catch per-structure failures and warn rather than raising, so the
    warnings are often the only record that part of a run did not succeed.

    Parameters
    ----------
    counts
        Number of times each (category, message) pair was emitted.

    Returns
    -------
    dict[str, Any]
        Total and per-message warning counts, truncated to the most common messages.
    """
    most_common = counts.most_common(MAX_UNIQUE_WARNINGS)
    return {
        "total": sum(counts.values()),
        "n_unique": len(counts),
        "truncated": len(counts) > len(most_common),
        "messages": [
            {"category": category, "message": message, "count": count}
            for (category, message), count in most_common
        ],
    }


def format_warning(category: str, message: str) -> tuple[str, str]:
    """
    Normalise a warning for counting.

    Parameters
    ----------
    category
        Name of the warning category.
    message
        Warning message.

    Returns
    -------
    tuple[str, str]
        Category and truncated message, suitable as a counter key.
    """
    text = " ".join(str(message).split())
    if len(text) > MAX_WARNING_CHARS:
        text = f"{text[:MAX_WARNING_CHARS]}..."
    return category, text


def format_error(
    exc_type: str, message: str, traceback: str | None = None
) -> dict[str, str]:
    """
    Describe an exception that ended a run.

    Parameters
    ----------
    exc_type
        Name of the exception class.
    message
        Exception message.
    traceback
        Formatted traceback. Default is `None`.

    Returns
    -------
    dict[str, str]
        Truncated description of the error.
    """
    error = {"type": exc_type, "message": str(message)[:MAX_ERROR_CHARS]}
    if traceback:
        error["traceback"] = traceback[-MAX_TRACEBACK_CHARS:]
    return error


def log_path(out_path: Path | str, model_name: str) -> Path:
    """
    Get the path to a model's run log.

    Parameters
    ----------
    out_path
        Output directory of the benchmark, containing one directory per model.
    model_name
        Name of the model.

    Returns
    -------
    Path
        Path to the model's run log.
    """
    return Path(out_path) / model_name / RUN_LOG_NAME


def read_runs(out_path: Path | str, model_name: str) -> list[dict[str, Any]]:
    """
    Read every run logged for a model, oldest first.

    Parameters
    ----------
    out_path
        Output directory of the benchmark, containing one directory per model.
    model_name
        Name of the model.

    Returns
    -------
    list[dict[str, Any]]
        Logged runs. Empty if the model has never been run.
    """
    return _read_runs(log_path(out_path, model_name))


def latest_run(out_path: Path | str, model_name: str) -> dict[str, Any] | None:
    """
    Read the most recent run logged for a model.

    Parameters
    ----------
    out_path
        Output directory of the benchmark, containing one directory per model.
    model_name
        Name of the model.

    Returns
    -------
    dict[str, Any] | None
        Most recent run, or `None` if the model has never been run.
    """
    runs = read_runs(out_path, model_name)
    return runs[-1] if runs else None


def collect_runs(
    category: str = "*",
    test: str = "*",
    models: str | None = None,
    models_file: Path | str | None = None,
) -> dict[str, dict[str, dict[str, Any] | None]]:
    """
    Read the most recent run of each model, for each matching benchmark.

    Models that have been run but are no longer defined are included, so results
    left behind by a since-removed model are still visible.

    Parameters
    ----------
    category
        Category of benchmarks to read. Default is `*`, corresponding to all
        categories.
    test
        Benchmark to read. Default is `*`, corresponding to all benchmarks.
    models
        Models to read, in a comma-separated list. Default is `None`, corresponding
        to all defined models.
    models_file
        Path to model definitions YAML file. Default is `None`, using models.yml.

    Returns
    -------
    dict[str, dict[str, dict[str, Any] | None]]
        Most recent run of each model, keyed by benchmark then model. `None` where
        a model has never been run.
    """
    from ml_peg.models.get_models import get_model_names

    defined = get_model_names(models, models_file)
    selected = (
        {name.strip() for name in models.split(",")}
        if isinstance(models, str)
        else None
    )

    runs: dict[str, dict[str, dict[str, Any] | None]] = {}
    for script in sorted(CALCS_ROOT.glob(f"{category}/{test}/calc_*.py")):
        benchmark = benchmark_id(script)
        if benchmark is None:
            continue

        out_path = script.parent / "outputs"
        logged = sorted(path.parent.name for path in out_path.glob(f"*/{RUN_LOG_NAME}"))
        if selected is not None:
            logged = [name for name in logged if name in selected]

        # Defined models first, so the table reads in the order models are listed
        names = dict.fromkeys((*defined, *logged))
        runs[benchmark] = {name: latest_run(out_path, name) for name in names}

    return runs


def is_stale(run: dict[str, Any]) -> bool:
    """
    Check whether a model's definition has changed since it was run.

    Parameters
    ----------
    run
        Logged run to check.

    Returns
    -------
    bool
        Whether the model is still defined as it was when the run was logged.
    """
    logged = (run.get("model_config") or {}).get("config_sha256")
    if logged is None:
        return False
    return logged != model_config(run["model"]).get("config_sha256")


def _read_runs(path: Path) -> list[dict[str, Any]]:
    """
    Read previously logged runs.

    Parameters
    ----------
    path
        Path to the run log.

    Returns
    -------
    list[dict[str, Any]]
        Logged runs, oldest first. Empty if the log is missing or unreadable.
    """
    try:
        log = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return []
    runs = log.get("runs", []) if isinstance(log, dict) else []
    return runs if isinstance(runs, list) else []


def _write_runs(path: Path, runs: list[dict[str, Any]]) -> None:
    """
    Write runs to a log, replacing it atomically.

    A partially written log would be unreadable by analysis, so the replacement is
    staged in the same directory and moved into place.

    Parameters
    ----------
    path
        Path to the run log.
    runs
        Logged runs, oldest first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    log = {"schema_version": SCHEMA_VERSION, "runs": runs}
    with NamedTemporaryFile(
        "w", encoding="utf8", dir=path.parent, delete=False
    ) as file:
        json.dump(log, file, indent=2, default=str)
        temporary = Path(file.name)
    temporary.replace(path)


def start_run(
    out_path: Path | str,
    model_name: str,
    benchmark: str | None = None,
    params: dict[str, Any] | None = None,
) -> Path:
    """
    Record the start of a calculation.

    The run is logged as "running" before the calculation begins, so a run killed
    without warning, such as by a scheduler timeout, is still distinguishable from
    one that was never started.

    Parameters
    ----------
    out_path
        Output directory of the benchmark, containing one directory per model.
    model_name
        Name of the model being run.
    benchmark
        Identifier of the benchmark. Default is `None`, deriving it from `out_path`.
    params
        Parameters distinguishing this run from others of the same benchmark and
        model, such as a pressure index. Default is `None`.

    Returns
    -------
    Path
        Path to the run log, to be passed to `end_run`.
    """
    out_path = Path(out_path)
    path = log_path(out_path, model_name)
    runs = _read_runs(path)
    runs.append(
        {
            "benchmark": benchmark if benchmark else benchmark_id(out_path.parent),
            "model": model_name,
            "params": params or {},
            "status": "running",
            "started": _timestamp(),
            "finished": None,
            "duration_s": None,
            "environment": _static_environment(),
            "model_config": model_config(model_name),
        }
    )
    _write_runs(path, runs)
    return path


def end_run(
    path: Path,
    status: Status,
    error: dict[str, str] | None = None,
    warning_counts: Counter[tuple[str, str]] | None = None,
) -> None:
    """
    Record the end of a calculation.

    Updates the run started by `start_run`, so a failure during teardown can replace
    an already recorded outcome.

    Parameters
    ----------
    path
        Path to the run log, as returned by `start_run`.
    status
        Outcome of the run.
    error
        Description of the exception that ended the run, from `format_error`.
        Default is `None`.
    warning_counts
        Number of times each (category, message) pair was warned. Default is `None`.
    """
    runs = _read_runs(path)
    if not runs:
        warnings.warn(f"No run to finalise in {path}", stacklevel=2)
        return

    run = runs[-1]
    finished = datetime.now(timezone.utc)
    run["status"] = status
    run["finished"] = finished.isoformat(timespec="seconds")
    try:
        started = datetime.fromisoformat(run["started"])
        run["duration_s"] = round((finished - started).total_seconds(), 1)
    except (TypeError, ValueError):
        run["duration_s"] = None
    if error:
        run["error"] = error
    if warning_counts is not None:
        run["warnings"] = summarise_warnings(warning_counts)
    if device := _device():
        run["environment"]["device"] = device

    _write_runs(path, runs)


@contextmanager
def log_run(
    out_path: Path | str,
    model_name: str,
    benchmark: str | None = None,
    params: dict[str, Any] | None = None,
) -> Iterator[None]:
    """
    Log a calculation that pytest does not run directly, such as a zntrack node.

    Warnings raised within the block are captured alongside the outcome.

    Parameters
    ----------
    out_path
        Output directory of the benchmark, containing one directory per model.
    model_name
        Name of the model being run.
    benchmark
        Identifier of the benchmark. Default is `None`, deriving it from `out_path`.
    params
        Parameters distinguishing this run from others of the same benchmark and
        model. Default is `None`.

    Yields
    ------
    None
        Yielded while the calculation runs.
    """
    import traceback as traceback_module

    path = start_run(out_path, model_name, benchmark, params)
    counts: Counter[tuple[str, str]] = Counter()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            yield
        except BaseException as exc:
            counts.update(
                format_warning(warning.category.__name__, warning.message)
                for warning in recorded
            )
            end_run(
                path,
                "failed",
                format_error(
                    type(exc).__name__, str(exc), traceback_module.format_exc()
                ),
                counts,
            )
            raise
        counts.update(
            format_warning(warning.category.__name__, warning.message)
            for warning in recorded
        )

    # Re-emit, as recording them above suppresses them
    for warning in recorded:
        warnings.warn_explicit(
            warning.message,
            warning.category,
            warning.filename,
            warning.lineno,
        )

    end_run(path, "completed", warning_counts=counts)
