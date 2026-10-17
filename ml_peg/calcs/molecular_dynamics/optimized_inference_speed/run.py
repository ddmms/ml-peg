"""Run optimised inference containers and measure them from outside."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from warnings import warn

import psutil
from yaml import safe_load

from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.build import (
    DEFAULT_CACHE,
    check_labels,
    inspect_labels,
    parse_def_labels,
)
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.contract import (
    CONTAINER_MODELS_FILE,
    CONTRIB_ROOT,
    RECORD_FILE,
    RESULT_FILE,
    build_invocation,
    format_run_id,
)
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.stage import (
    sha256_file,
)

SAMPLE_INTERVAL_S = 0.1


def load_container_models(path: Path = CONTAINER_MODELS_FILE) -> dict[str, dict]:
    """
    Load container metadata from the benchmark's ``models.yml``.

    Parameters
    ----------
    path
        Path to ``models.yml``.

    Returns
    -------
    dict[str, dict]
        Container name to metadata.
    """
    with path.open(encoding="utf8") as file:
        return safe_load(file)


@dataclasses.dataclass
class PlannedRun:
    """A single container run."""

    model: str
    precision: str
    material: str
    n_atoms: int
    repeat: int
    validate: bool

    @property
    def run_id(self) -> str:
        """
        Get the run ID.

        Returns
        -------
        str
            Run ID.
        """
        return format_run_id(self.material, self.n_atoms, self.repeat)

    def run_dir(self, campaign: Path) -> Path:
        """
        Get the output directory of the run.

        Parameters
        ----------
        campaign
            Campaign directory.

        Returns
        -------
        Path
            Output directory.
        """
        return campaign / "out" / self.model / self.precision / self.run_id


def plan_runs(
    spec: dict,
    precisions: dict[str, Sequence[str]],
    repeats: int,
    seed: int = 0,
) -> list[PlannedRun]:
    """
    Plan the order of runs in a campaign.

    Models and the repeats at each size are shuffled; sizes ascend within each
    material so that larger sizes can be skipped after an out-of-memory result.

    Parameters
    ----------
    spec
        Campaign spec.
    precisions
        Container name to the precision modes to run.
    repeats
        Number of repeats of each size.
    seed
        Seed for shuffling.

    Returns
    -------
    list[PlannedRun]
        Runs in launch order.
    """
    rng = random.Random(seed)
    models = sorted(precisions)
    rng.shuffle(models)
    runs = []
    for model in models:
        for precision in precisions[model]:
            for system in spec["systems"]:
                for size in sorted(system["sizes"], key=lambda s: s["n_atoms"]):
                    order = list(range(1, repeats + 1))
                    rng.shuffle(order)
                    runs.extend(
                        PlannedRun(
                            model=model,
                            precision=precision,
                            material=system["material"],
                            n_atoms=size["n_atoms"],
                            repeat=repeat,
                            validate=size.get("validate", False),
                        )
                        for repeat in order
                    )
    return runs


def _process_tree(pid: int) -> list[psutil.Process]:
    """
    Get a process and all its descendants.

    Parameters
    ----------
    pid
        Root process ID.

    Returns
    -------
    list[psutil.Process]
        Processes in the tree that still exist.
    """
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def _init_nvml(gpu_index: int) -> Any | None:
    """
    Initialise NVML and get a device handle.

    Parameters
    ----------
    gpu_index
        Index of the GPU.

    Returns
    -------
    Any | None
        Device handle, or None if NVML is unavailable.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception:  # noqa: BLE001 - any NVML failure means no GPU sampling
        return None


def _peak(samples: list[dict], name: str) -> int | None:
    """
    Get the largest value of a sampled quantity.

    Parameters
    ----------
    samples
        Samples to search.
    name
        Quantity to find the peak of.

    Returns
    -------
    int | None
        Peak value, or None if the quantity was never sampled.
    """
    values = [s[name] for s in samples if s[name] is not None]
    return max(values) if values else None


class ResourceSampler(threading.Thread):
    """
    Sample GPU and host memory of a process tree at a fixed interval.

    GPU memory is taken per process when NVML reports the tree's processes, and
    otherwise as device used memory minus an idle baseline.

    Parameters
    ----------
    gpu_index
        Index of the GPU to sample.
    interval
        Sampling interval in seconds.
    """

    def __init__(self, gpu_index: int = 0, interval: float = SAMPLE_INTERVAL_S):
        """
        Initialise sampler and take the idle GPU baseline.

        Parameters
        ----------
        gpu_index
            Index of the GPU to sample.
        interval
            Sampling interval in seconds.
        """
        super().__init__(daemon=True)
        self.interval = interval
        self.handle = _init_nvml(gpu_index)
        self.baseline = self._device_used()
        self.pid: int | None = None
        self.samples: list[dict[str, float | None]] = []
        self._stop_event = threading.Event()

    def _device_used(self) -> int | None:
        """
        Get device used memory.

        Returns
        -------
        int | None
            Used memory in bytes, or None without NVML.
        """
        if self.handle is None:
            return None
        import pynvml

        return int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)

    def _process_gpu(self, pids: set[int]) -> int | None:
        """
        Sum GPU memory used by the given processes.

        Parameters
        ----------
        pids
            Host process IDs.

        Returns
        -------
        int | None
            Memory in bytes, or None if NVML reports none of the processes.
        """
        if self.handle is None:
            return None
        import pynvml

        try:
            procs = [
                *pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle),
                *pynvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle),
            ]
        except pynvml.NVMLError:
            return None
        used = [
            proc.usedGpuMemory
            for proc in procs
            if proc.pid in pids and proc.usedGpuMemory is not None
        ]
        return int(sum(used)) if used else None

    def sample(self) -> None:
        """Take one sample of the process tree."""
        procs = _process_tree(self.pid) if self.pid else []
        rss = 0
        for proc in procs:
            try:
                rss += proc.memory_info().rss
            except psutil.Error:
                continue
        device = self._device_used()
        self.samples.append(
            {
                "t": time.time(),
                "host_rss": rss,
                "gpu_process": self._process_gpu({proc.pid for proc in procs}),
                "gpu_device": (
                    None if device is None else max(0, device - self.baseline)
                ),
            }
        )

    def run(self) -> None:
        """Sample until stopped."""
        while not self._stop_event.is_set():
            self.sample()
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        """Stop sampling, taking a final sample."""
        self._stop_event.set()
        self.join()

    def summary(self, window: tuple[float, float] | None = None) -> dict[str, Any]:
        """
        Summarise peak memory over the whole run and within `window`.

        Parameters
        ----------
        window
            Start and end of the measured phase, as Unix timestamps.

        Returns
        -------
        dict[str, Any]
            Peak memory and the GPU memory method used.
        """
        if self.handle is None:
            method = "unavailable"
        elif any(s["gpu_process"] is not None for s in self.samples):
            method = "per_process"
        else:
            method = "device_minus_baseline"
        key = "gpu_process" if method == "per_process" else "gpu_device"
        windowed = []
        if window is not None:
            windowed = [s for s in self.samples if window[0] <= s["t"] <= window[1]]
        return {
            "gpu_method": method,
            "gpu_baseline_bytes": self.baseline,
            "gpu_peak_bytes": _peak(self.samples, key),
            "gpu_peak_measured_bytes": _peak(windowed, key),
            "host_rss_peak_bytes": _peak(self.samples, "host_rss"),
            "host_rss_peak_measured_bytes": _peak(windowed, "host_rss"),
            "n_samples": len(self.samples),
            "n_samples_measured": len(windowed),
        }


def collect_hardware(gpu_index: int = 0) -> dict[str, Any]:
    """
    Record the hardware and software environment of the host.

    Parameters
    ----------
    gpu_index
        Index of the GPU used for runs.

    Returns
    -------
    dict[str, Any]
        Hardware record, including a ``hardware_id``.
    """
    cpu_model = platform.processor() or None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    try:
        apptainer = subprocess.run(
            ["apptainer", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        apptainer = None

    record: dict[str, Any] = {
        "hostname": platform.node(),
        "cpu_model": cpu_model,
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "host_ram_bytes": psutil.virtual_memory().total,
        "apptainer_version": apptainer,
        "gpu": None,
    }
    handle = _init_nvml(gpu_index)
    if handle is not None:
        import pynvml

        def clock(kind: int, getter: Any) -> int | None:
            """
            Get a GPU clock, if NVML reports it.

            Parameters
            ----------
            kind
                NVML clock type.
            getter
                NVML function returning the clock.

            Returns
            -------
            int | None
                Clock in MHz.
            """
            try:
                return int(getter(handle, kind))
            except pynvml.NVMLError:
                return None

        cuda = pynvml.nvmlSystemGetCudaDriverVersion()
        record["gpu"] = {
            "index": gpu_index,
            "name": pynvml.nvmlDeviceGetName(handle),
            "memory_total_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "cuda_version": f"{cuda // 1000}.{cuda % 1000 // 10}",
            "clocks_mhz": {
                "sm_max": clock(pynvml.NVML_CLOCK_SM, pynvml.nvmlDeviceGetMaxClockInfo),
                "mem_max": clock(
                    pynvml.NVML_CLOCK_MEM, pynvml.nvmlDeviceGetMaxClockInfo
                ),
                "sm_current": clock(
                    pynvml.NVML_CLOCK_SM, pynvml.nvmlDeviceGetClockInfo
                ),
                "mem_current": clock(
                    pynvml.NVML_CLOCK_MEM, pynvml.nvmlDeviceGetClockInfo
                ),
            },
        }
    gpu_name = record["gpu"]["name"] if record["gpu"] else "nogpu"
    slug = re.sub(r"[^a-z0-9]+", "-", f"{record['hostname']}-{gpu_name}".lower())
    record["hardware_id"] = slug.strip("-")
    return record


def launch(
    command: list[str],
    run_dir: Path,
    timeout: float,
    env: dict[str, str] | None = None,
    gpu_index: int = 0,
) -> dict[str, Any]:
    """
    Launch one run, sampling memory until it exits or times out.

    Parameters
    ----------
    command
        Command to run.
    run_dir
        Output directory; stdout and stderr are written here.
    timeout
        Timeout in seconds.
    env
        Environment for the process. Default is the current environment.
    gpu_index
        Index of the GPU to sample.

    Returns
    -------
    dict[str, Any]
        Exit code, wall time, timeout flag and the sampler.
    """
    sampler = ResourceSampler(gpu_index=gpu_index)
    with (
        (run_dir / "stdout.log").open("w") as stdout,
        (run_dir / "stderr.log").open("w") as stderr,
    ):
        launched = time.time()
        proc = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, env=env, start_new_session=True
        )
        sampler.pid = proc.pid
        sampler.start()
        timed_out = False
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            exit_code = proc.wait()
        exited = time.time()
        sampler.stop()
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "launch_unix": launched,
        "exit_unix": exited,
        "wall_time_s": exited - launched,
        "sampler": sampler,
    }


def _measured_window(result: dict | None) -> tuple[float, float] | None:
    """
    Get the measured phase from a container result.

    Parameters
    ----------
    result
        Parsed ``inference.json``, if any.

    Returns
    -------
    tuple[float, float] | None
        Start and end of the measured phase, if reported.
    """
    try:
        stamps = result["timestamps_unix"]
        return float(stamps["warmup_end"]), float(stamps["measure_end"])
    except (KeyError, TypeError, ValueError):
        return None


def _resolve_container(
    model: str,
    runtime: str,
    cache_dir: Path,
    contrib_dir: Path,
    container_models: dict[str, dict],
) -> dict[str, Any]:
    """
    Get the labels, checksum and launch details of a container.

    Parameters
    ----------
    model
        Container name.
    runtime
        ``apptainer`` or ``local``.
    cache_dir
        Directory containing built images.
    contrib_dir
        Directory containing container definitions.
    container_models
        Container metadata from ``models.yml``.

    Returns
    -------
    dict[str, Any]
        Container details.
    """
    if runtime == "apptainer":
        sif = cache_dir / f"{model}.sif"
        if not sif.is_file():
            raise FileNotFoundError(f"{sif} not found; build it first.")
        labels = inspect_labels(sif)
        details = {"sif": sif, "sif_sha256": sha256_file(sif)}
    else:
        model_dir = contrib_dir / model
        labels = parse_def_labels(model_dir / "container.def")
        entry = container_models.get(model, {}).get("local_entry")
        if not entry:
            raise ValueError(f"No local_entry for '{model}' in models.yml.")
        command = []
        for part in entry:
            part = part.replace("{python}", sys.executable)
            if (model_dir / part).exists():
                part = str(model_dir / part)
            command.append(part)
        details = {"local_entry": command, "sif_sha256": None}
    problems = check_labels(labels)
    if problems:
        raise ValueError(f"Container '{model}': {'; '.join(problems)}")
    details["labels"] = labels
    details["precisions"] = labels["mlpeg.precisions"].split(",")
    return details


def _write_json(path: Path, data: dict) -> None:
    """
    Write a dictionary as indented JSON.

    Parameters
    ----------
    path
        File to write.
    data
        Data to write.
    """
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")


def run_campaign(
    campaign: Path,
    models: Sequence[str] | None = None,
    runtime: str = "apptainer",
    cache_dir: Path = DEFAULT_CACHE,
    contrib_dir: Path = CONTRIB_ROOT,
    precisions: Sequence[str] | None = None,
    repeats: int | None = None,
    timeout: float | None = None,
    container_env: Sequence[str] = (),
    seed: int = 0,
    gpu_index: int = 0,
) -> list[dict[str, Any]]:
    """
    Run every container on every staged structure.

    Parameters
    ----------
    campaign
        Campaign directory created by staging.
    models
        Container names. Default is all containers in ``models.yml``.
    runtime
        ``apptainer`` runs the contract invocation; ``local`` runs each
        container's ``local_entry`` directly, for testing without Apptainer.
    cache_dir
        Directory containing built images.
    contrib_dir
        Directory containing container definitions.
    precisions
        Only run these precision modes. Default is every mode a container declares.
    repeats
        Number of repeats. Default is taken from the spec.
    timeout
        Timeout per run in seconds. Default is taken from the spec.
    container_env
        Extra ``KEY=VALUE`` environment variables passed to the containers.
    seed
        Seed for shuffling the run order.
    gpu_index
        Index of the GPU to sample.

    Returns
    -------
    list[dict[str, Any]]
        Records of all runs.
    """
    if runtime not in ("apptainer", "local"):
        raise ValueError(f"Unknown runtime '{runtime}'")
    campaign = campaign.resolve()
    spec_path = campaign / "in" / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    execution = spec.get("execution", {})
    repeats = repeats or execution.get("repeats", 3)
    timeout = timeout or execution.get("run_timeout_seconds", 1800)

    hardware = collect_hardware(gpu_index)
    (campaign / "out").mkdir(parents=True, exist_ok=True)
    _write_json(campaign / "out" / "hardware.json", hardware)
    if not spec["campaign"].get("hardware_id"):
        spec["campaign"]["hardware_id"] = hardware["hardware_id"]
        _write_json(spec_path, spec)

    container_models = load_container_models()
    models = list(models) if models else list(container_models)
    containers = {
        model: _resolve_container(
            model, runtime, cache_dir, contrib_dir, container_models
        )
        for model in models
    }
    run_precisions = {
        model: [p for p in details["precisions"] if not precisions or p in precisions]
        for model, details in containers.items()
    }

    oom_at: dict[tuple[str, str, str], int] = {}
    records = []
    for planned in plan_runs(spec, run_precisions, repeats, seed):
        container = containers[planned.model]
        run_dir = planned.run_dir(campaign)
        run_dir.mkdir(parents=True, exist_ok=True)
        for stale in (RESULT_FILE, RECORD_FILE):
            (run_dir / stale).unlink(missing_ok=True)
        record: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": planned.run_id,
            "model": planned.model,
            "mlpeg_model": container["labels"]["mlpeg.model"],
            "precision": planned.precision,
            "material": planned.material,
            "n_atoms": planned.n_atoms,
            "repeat": planned.repeat,
            "validate": planned.validate,
            "runtime": runtime,
            "sif_sha256": container["sif_sha256"],
            "labels": container["labels"],
            "hardware": hardware,
        }

        group = (planned.model, planned.precision, planned.material)
        if group in oom_at and planned.n_atoms > oom_at[group]:
            record.update(
                status="skipped",
                error=f"skipped after oom at {oom_at[group]} atoms",
                exit_code=None,
                wall_time_s=None,
            )
            _write_json(run_dir / RECORD_FILE, record)
            records.append(record)
            continue

        if runtime == "apptainer":
            command = build_invocation(
                container["sif"],
                campaign / "in",
                run_dir,
                planned.run_id,
                planned.precision,
                container_env,
            )
            env = None
        else:
            command = [
                *container["local_entry"],
                "--spec",
                str(spec_path),
                "--run-id",
                planned.run_id,
                "--precision",
                planned.precision,
                "--out",
                str(run_dir),
            ]
            env = {
                "PATH": os.environ.get("PATH", ""),
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "8",
            }
            env.update(item.split("=", 1) for item in container_env)

        print(f"Running {planned.model} {planned.precision} {planned.run_id}")
        launched = launch(command, run_dir, timeout, env=env, gpu_index=gpu_index)

        result = None
        if (run_dir / RESULT_FILE).is_file():
            try:
                result = json.loads((run_dir / RESULT_FILE).read_text())
            except json.JSONDecodeError:
                result = None
        if launched["timed_out"]:
            status, error = "timeout", f"timed out after {timeout} s"
        elif launched["exit_code"] != 0:
            status, error = "error", f"exit code {launched['exit_code']}"
        elif result is None:
            status, error = "error", f"missing or unreadable {RESULT_FILE}"
        else:
            status, error = str(result.get("status")), result.get("error")
        if status == "oom":
            oom_at[group] = min(oom_at.get(group, planned.n_atoms), planned.n_atoms)

        sampler = launched.pop("sampler")
        record.update(
            status=status,
            error=error,
            command=command,
            memory=sampler.summary(_measured_window(result)),
            **launched,
        )
        _write_json(run_dir / RECORD_FILE, record)
        records.append(record)
        if status in ("error", "timeout"):
            warn(f"{planned.model} {planned.run_id}: {error}", stacklevel=2)
    return records
