"""Define classes for all models."""

# ruff: noqa: D101, D102, F401

from __future__ import annotations

import copy
import dataclasses
from functools import cache, wraps
import hashlib
import importlib
from importlib.util import find_spec
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Literal, get_args
from warnings import warn

from mlipx import GenericASECalculator as MlipxGenericASECalc
from mlipx.nodes.generic_ase import Device

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ase.calculators.calculator import Calculator
    from ase.calculators.mixing import SumCalculator

Precision = Literal["low", "high"]
DeviceType = Literal["auto", "cpu", "cuda"]

#: Keys read from a model's `dispersion_kwargs`. `label` is used to name
#: dispersion-corrected models, the rest by `SumCalc.add_d3_calculator`.
DISPERSION_KWARGS = frozenset(
    {"device", "damping", "xc", "dtype", "cutoff", "label"},
)

#: Calculators that each `class_name` forwards its `**kwargs` to, whose arguments
#: are therefore valid too.
FORWARDED_KWARGS = {
    "AIMNet2ASE": (("aimnet.calculators", "AIMNet2Calculator"),),
    "mace_mp": (("mace.calculators", "MACECalculator"),),
    "mace_off": (("mace.calculators", "MACECalculator"),),
    "mace_omol": (("mace.calculators", "MACECalculator"),),
    "mace_polar": (("mace.calculators", "MACECalculator"),),
    "grace_fm": (("tensorpotential.calculator", "TPCalculator"),),
    "MatterSimCalculator": (
        ("mattersim.forcefield.potential", "Potential.from_checkpoint"),
    ),
}

#: Keyword arguments read from `**kwargs`, so absent from every signature.
MACE_KWARGS = {"head", "compute_atomic_stresses"}
PASSTHROUGH_KWARGS = {
    "mace_mp": MACE_KWARGS,
    "mace_off": MACE_KWARGS,
    "mace_omol": MACE_KWARGS,
    "mace_polar": MACE_KWARGS,
    "MACECalculator": MACE_KWARGS,
    "grace_fm": {"cutoff"},
}


def _named_parameters(target: Callable) -> set[str]:
    """
    Get the keyword arguments a callable accepts by name.

    Parameters
    ----------
    target
        Callable to inspect.

    Returns
    -------
    set[str]
        Parameter names, excluding any `**kwargs`.
    """
    return {
        name
        for name, parameter in inspect.signature(target).parameters.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD and name != "self"
    }


def _takes_var_keyword(target: Callable) -> bool:
    """
    Check whether a callable collects extra keyword arguments.

    Parameters
    ----------
    target
        Callable to inspect.

    Returns
    -------
    bool
        Whether the callable takes `**kwargs`.
    """
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(target).parameters.values()
    )


def accepted_kwargs(target: Callable, class_name: str) -> set[str]:
    """
    Get the keyword arguments a calculator accepts.

    Covers its own parameters, those of calculators it forwards `**kwargs` to, ASE
    class defaults, and arguments it reads from `**kwargs` itself.

    Parameters
    ----------
    target
        Calculator class or factory function the keyword arguments are passed to.
    class_name
        Registry `class_name`, used to look up what `target` forwards to.

    Returns
    -------
    set[str]
        Names of all accepted keyword arguments.
    """
    from ase.calculators.calculator import Calculator

    accepted = _named_parameters(target)

    targets = [target]
    for module, forwarded in FORWARDED_KWARGS.get(class_name, ()):
        try:
            targets.append(import_calculator(module, forwarded))
        except (ImportError, AttributeError):
            continue

    for forwarded_target in targets:
        if forwarded_target is not target:
            accepted |= _named_parameters(forwarded_target)

        # Anything else is only reached through `**kwargs`
        if not _takes_var_keyword(forwarded_target):
            continue

        # ASE calculators also take their class defaults as keyword arguments
        accepted |= set(getattr(forwarded_target, "default_parameters", None) or ())
        if isinstance(forwarded_target, type) and issubclass(
            forwarded_target, Calculator
        ):
            accepted |= _named_parameters(Calculator.__init__)

    return accepted | PASSTHROUGH_KWARGS.get(class_name, set())


def check_kwargs(
    kwargs: dict[str, Any],
    target: Callable,
    class_name: str,
    reserved: Sequence[str] = (),
) -> None:
    """
    Check that configured keyword arguments are accepted by the calculator.

    Most calculators end in `**kwargs`, which ASE stores without complaint, so a
    misnamed key would otherwise be ignored.

    Parameters
    ----------
    kwargs
        Keyword arguments set by the model's `kwargs` key.
    target
        Calculator class or factory function the keyword arguments are passed to.
    class_name
        Registry `class_name`, used to look up what `target` accepts indirectly.
    reserved
        Parameter names set by ML-PEG itself, which must not be overridden.

    Raises
    ------
    ValueError
        If a keyword argument is reserved, or is accepted by neither `target` nor
        anything it passes its own `**kwargs` on to.
    """
    check_reserved(kwargs, reserved)

    # Nothing to check against, such as a test double taking only `**kwargs`.
    if not _named_parameters(target) and not getattr(
        target, "default_parameters", None
    ):
        return

    accepted = accepted_kwargs(target, class_name)
    if unknown := sorted(set(kwargs) - accepted):
        raise ValueError(
            f"{', '.join(unknown)} would be ignored by {class_name}. Valid keyword "
            f"arguments are: {', '.join(sorted(accepted))}"
        )


def check_reserved(kwargs: dict[str, Any], reserved: Sequence[str]) -> None:
    """
    Check that keyword arguments set by ML-PEG are not also set by the model.

    Parameters
    ----------
    kwargs
        Keyword arguments set by the model's `kwargs` key.
    reserved
        Parameter names set by ML-PEG itself, which must not be overridden.

    Raises
    ------
    ValueError
        If a reserved keyword argument is set.
    """
    if overridden := sorted(set(kwargs) & set(reserved)):
        raise ValueError(
            f"{', '.join(overridden)} is set by ML-PEG, so cannot be set through "
            "model kwargs. Precision is set by the benchmark, or by overwrite_dtype"
        )


def import_calculator(module: str, class_name: str) -> Callable:
    """
    Import the calculator named by a model's registry entry.

    Parameters
    ----------
    module
        Module containing the calculator, as set by the registry's `module` key.
    class_name
        Calculator class or factory function, as set by the registry's `class_name` key.
        Dotted names are resolved in turn, e.g. `Potential.from_checkpoint`.

    Returns
    -------
    Callable
        Calculator class or factory function.
    """
    calculator = importlib.import_module(module)
    for attribute in class_name.split("."):
        calculator = getattr(calculator, attribute)

    return calculator


def split_kwargs(
    kwargs: dict[str, Any],
    targets: Sequence[Callable],
    reserved: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """
    Split keyword arguments between the callables that accept them.

    Some models are loaded in two stages, so each kwarg must be routed to the stage
    that takes it.

    Parameters
    ----------
    kwargs
        Keyword arguments to be split between `targets`.
    targets
        Callables that the keyword arguments will be passed to. Each key is routed to
        the first target that accepts it.
    reserved
        Parameter names set by ML-PEG itself, which must not be overridden.

    Returns
    -------
    list[dict[str, Any]]
        Keyword arguments for each target, in the same order as `targets`.

    Raises
    ------
    ValueError
        If a keyword argument is reserved, or is accepted by none of the targets.
    """
    check_reserved(kwargs, reserved)

    accepted = [
        accepted_kwargs(target, getattr(target, "__name__", "")) for target in targets
    ]
    split: list[dict[str, Any]] = [{} for _ in targets]

    for key, value in kwargs.items():
        match = next(
            (
                target_kwargs
                for names, target_kwargs in zip(accepted, split, strict=True)
                if key in names
            ),
            None,
        )
        if match is None:
            names = ", ".join(target.__name__ for target in targets)
            raise ValueError(f"'{key}' is not a valid keyword argument for {names}")
        match[key] = value

    return split


def warn_precision_unused(class_name: str, default_dtype: str | None) -> None:
    """
    Warn that `overwrite_dtype` cannot be applied to a calculator.

    Parameters
    ----------
    class_name
        Registry `class_name` of the calculator.
    default_dtype
        Value set by the model's `overwrite_dtype` key, if any.
    """
    if default_dtype is None:
        return

    warn(
        f"{class_name} has no dtype argument, so overwrite_dtype "
        f"'{default_dtype}' has no effect",
        stacklevel=3,
    )


def check_precision(precision: str) -> None:
    """
    Validate the precision argument.

    Parameters
    ----------
    precision
        Precision string to validate.

    Raises
    ------
    ValueError
        If the precision is not one of "low" or "high".
    """
    if precision not in get_args(Precision):
        raise ValueError(
            f"Invalid precision '{precision}'. Must be one of {get_args(Precision)}."
        )


def _patch_metatomic_nvalchemi_max_neighbors() -> None:
    """
    Make metatomic's CUDA neighbor-list call compatible with nvalchemi.

    ``metatomic-ase`` computes ``max_neighbors`` using ``cutoff**3``. For
    cutoffs above roughly 5 Å this produces a float, while nvalchemi requires
    an integer tensor dimension. PET-OAM uses a 10 Å cutoff.
    """
    import metatomic_ase._neighbors as metatomic_neighbors

    neighbor_list = metatomic_neighbors.nvalchemi_neighbor_list
    if getattr(neighbor_list, "_ml_peg_integer_max_neighbors", False):
        return

    @wraps(neighbor_list)
    def neighbor_list_with_integer_max_neighbors(*args: Any, **kwargs: Any) -> Any:
        """
        Convert nvalchemi's maximum-neighbor estimate to an integer.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the original neighbor-list function.
        **kwargs
            Keyword arguments forwarded to the original neighbor-list function.

        Returns
        -------
        Any
            The result from the original neighbor-list function.
        """
        max_neighbors = kwargs.get("max_neighbors")
        if max_neighbors is not None:
            kwargs["max_neighbors"] = int(max_neighbors)
        return neighbor_list(*args, **kwargs)

    neighbor_list_with_integer_max_neighbors._ml_peg_integer_max_neighbors = True
    metatomic_neighbors.nvalchemi_neighbor_list = (
        neighbor_list_with_integer_max_neighbors
    )


# Modules providing the GPU kernels of each nequip modifier, in order of preference:
_NEQUIP_KERNEL_PACKAGES = {
    "enable_OpenEquivariance": "openequivariance",
    "enable_CuEquivarianceContracter": "cuequivariance_ops_torch",
    "enable_TritonContracter": "triton",
}


# TODO: load the models from nequip.net once it serves the 0.2 packages, replacing this
# Zenodo download
def _download_nequip_package(url: str) -> str:
    """
    Download a NequIP package once and return the local path.

    Parameters
    ----------
    url
        URL of the ``.nequip.zip`` package, e.g. a Zenodo file ``content`` link.

    Returns
    -------
    str
        Path to the cached package.
    """
    import requests

    name = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = Path.home() / ".cache" / "ml_peg" / "nequip" / f"{name}.nequip.zip"
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            file.write(response.content)
        # concurrent first uses may download twice, so rename to keep only one:
        Path(file.name).replace(path)
    return str(path)


def _nequip_kernels(modifiers: set[str], device: str) -> list[str]:
    """
    List the installed GPU kernels a NequIP package can use on a device, fastest first.

    On CUDA, NequIP models use OpenEquivariance kernels if installed, and cuEquivariance
    then Triton for Allegro models.

    Parameters
    ----------
    modifiers
        Names of the modifiers the package supports.
    device
        Device the model will run on.

    Returns
    -------
    list[str]
        GPU kernel modifiers to try.
    """
    import torch

    cuda = torch.device(device).type == "cuda"
    return [
        kernel
        for kernel, package in _NEQUIP_KERNEL_PACKAGES.items()
        if cuda and kernel in modifiers and find_spec(package)
    ]


def _compile_nequip_package(
    model: str, device: str, modifier: str | None = None
) -> str:
    """
    Compile a NequIP package once per machine type and thread count, returning the file.

    Compiled code is specific to the CPU's instruction set or the GPU model.

    Parameters
    ----------
    model
        Path to the ``.nequip.zip`` package.
    device
        Device to compile for, which the compiled model must then be loaded on.
    modifier
        GPU kernel modifier to compile with, if any.

    Returns
    -------
    str
        Path to the cached compiled model.
    """
    from filelock import FileLock
    import nequip
    import torch

    env = dict(os.environ)
    if torch.device(device).type == "cuda":
        hardware = torch.cuda.get_device_name(device)
    else:
        hardware = f"{platform.machine()}-{torch.backends.cpu.get_cpu_capability()}"
        # TODO: nequip > 0.19.1 compiles scalar C++ unless ATEN_CPU_CAPABILITY is set,
        # so stop setting it here (but keep it in the key) once that is required
        env.setdefault("ATEN_CPU_CAPABILITY", "default")
        # compiled kernels keep the thread count they were compiled with, so key on it:
        env["OMP_NUM_THREADS"] = threads = str(torch.get_num_threads())
        hardware += f"-simd_{env['ATEN_CPU_CAPABILITY']}-threads{threads}"
    if sys.platform == "linux":
        # torch compiles with -march=native here, so also key on the exact host CPU
        cpu = re.findall(
            r"^(?:model name|flags|Features|CPU part)\s*:.*$",
            Path("/proc/cpuinfo").read_text(),
            re.M,
        )
        cpu_id = hashlib.sha256("\n".join(sorted(set(cpu))).encode()).hexdigest()
        hardware += f"-{cpu_id[:12]}"

    model = str(Path(model).absolute())  # as nequip-compile runs in the cache directory
    digest = hashlib.sha256(Path(model).read_bytes()).hexdigest()
    key = [digest[:16], device, hardware]
    if modifier:
        kernels = importlib.import_module(_NEQUIP_KERNEL_PACKAGES[modifier])
        key += [modifier, f"{kernels.__name__}{kernels.__version__}"]

    key += [f"torch{torch.__version__}", f"nequip{nequip.__version__}"]
    name = re.sub(r"[^\w.+-]", "_", "-".join(key))
    suffix = ".nequip.pt2"
    path = Path.home() / ".cache" / "ml_peg" / "nequip" / "compiled" / f"{name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    # The lock stops parallel workers compiling the same model, while renames avoid
    # duplicate compilations where file locks are not shared (e.g. across nodes)
    with FileLock(f"{path}.lock", timeout=1000):
        if not path.exists():
            fd, partial = tempfile.mkstemp(suffix=suffix, dir=path.parent)
            os.close(fd)
            command = [sys.executable, "-m", "nequip.scripts.compile", model, partial]
            command += ["--mode", "aotinductor", "--device", device, "--target", "ase"]
            if modifier:
                command += ["--modifiers", modifier]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=path.parent,
                )
                os.replace(partial, path)
            finally:
                Path(partial).unlink(missing_ok=True)
    return str(path)


# TODO: nequip > 0.19.1 reuses one importer per package file, so repeated loads no
# longer leak; when required, drop this cache and the JSON options/copies it needs
@cache
def _load_nequip_calculator(
    model: str, device: str, compile: bool, options: str
) -> Calculator:
    """
    Load a NequIP package once per process, compiled if requested.

    On CUDA, compiled and eager models alike use the first GPU kernels that load.

    Parameters
    ----------
    model
        Path to the ``.nequip.zip`` package.
    device
        Device to load the model on.
    compile
        Whether to compile the model, trying each GPU kernel and then none. Each
        attempt that fails is skipped with a warning, ending with the eager model.
    options
        JSON-encoded keyword arguments for the ``NequIPCalculator`` constructors.

    Returns
    -------
    Calculator
        Loaded NequIP ASE calculator.
    """
    from nequip.integrations.ase import NequIPCalculator
    from nequip.model.modify_utils import get_all_modifiers, modify
    from nequip.utils.dtype import torch_default_dtype
    import torch

    class MLPEGNequIPCalculator(NequIPCalculator):
        """
        ``NequIPCalculator`` with float64 inputs.

        NequIP keeps its data (positions, cell) in float64 and runs the model in the
        package's ``model_dtype`` (typically float32). ``NequIPCalculator`` builds the
        data with ``torch.get_default_dtype()``, relying on NequIP's first load making
        float64 the process-wide default, which ML-PEG restores after loading.
        """

        calculate = torch_default_dtype(torch.float64)(NequIPCalculator.calculate)

    kwargs = json.loads(options)
    eager = MLPEGNequIPCalculator._from_saved_model(model, device=device, **kwargs)
    kernels = _nequip_kernels(set(get_all_modifiers(eager.model)), device)
    for kernel in (
        [*kernels, None] if compile else ()
    ):  # AOTInductor without GPU kernels comes last
        try:
            if kernel:  # GPU kernel libraries must be imported before loading the model
                importlib.import_module(_NEQUIP_KERNEL_PACKAGES[kernel])
            path = _compile_nequip_package(model, device, kernel)
            return MLPEGNequIPCalculator.from_compiled_model(
                path, device=device, **kwargs
            )
        # compiling needs a C++ compiler, and only changes speed:
        except Exception as error:
            details = getattr(error, "stderr", None) or repr(error)
            warn(
                f"Skipping the compiled NequIP model with "
                f"{kernel or 'no GPU kernels'}: {details[-2000:]}",
                stacklevel=2,
            )

    # eager models use the first GPU kernels that load too:
    for kernel in kernels:
        try:
            importlib.import_module(_NEQUIP_KERNEL_PACKAGES[kernel])
            # swapped-in modules may be built on the CPU, and start in training mode,
            # where e.g. Allegro's Triton contracter is not used (hence .eval()):
            eager.model = modify(eager.model, [{"modifier": kernel}]).to(device).eval()
            break
        except Exception as error:
            warn(
                f"Skipping the eager NequIP model with {kernel}: {error!r}",
                stacklevel=2,
            )
    if compile:
        warn(
            "Using the eager (non-compiled) NequIP model (set `compile: false` in "
            "`kwargs` to skip compiling)",
            stacklevel=2,
        )
    return eager


@dataclasses.dataclass(kw_only=True)
class SumCalc:
    """
    Base class that tracks whether a model already includes dispersion corrections.

    ``add_d3_calculator`` only wraps calculators with an explicit TorchDFTD3
    correction when ``trained_on_dispersion`` is ``False``; otherwise the original
    calculator is returned untouched.
    """

    trained_on_dispersion: bool = False
    dispersion_kwargs: dict = dataclasses.field(default_factory=dict)

    def add_d3_calculator(self, calcs) -> Calculator | SumCalculator:
        """
        Add dispersion corrections to calculator(s).

        Parameters
        ----------
        calcs
            Calculator, or list of calculators, to add dispersion corrections to via a
            SumCalculator.

        Returns
        -------
        SumCalculator | Calculator
            Calculator(s) with dispersion corrections added, or the original calculator
            when the model is already trained with dispersion corrections.
        """
        if self.trained_on_dispersion:
            return calcs
        from ase import units
        from ase.calculators.mixing import SumCalculator
        import torch
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

        if not isinstance(calcs, list):
            calcs = [calcs]

        d3_calc = TorchDFTD3Calculator(
            device=self.dispersion_kwargs.get("device", "cpu"),
            damping=self.dispersion_kwargs.get("damping", "bj"),
            xc=self.dispersion_kwargs.get("xc", "pbe"),
            dtype=getattr(torch, self.dispersion_kwargs.get("dtype", "float32")),
            cutoff=self.dispersion_kwargs.get("cutoff", 40.0 * units.Bohr),
        )
        calcs.append(d3_calc)

        return SumCalculator(calcs)


@dataclasses.dataclass(kw_only=True)
class GenericASECalc(SumCalc, MlipxGenericASECalc):
    """Data class for generic ASE calculators."""

    default_dtype: str | None = None

    def load_calculator(self, **kwargs) -> Calculator:
        """
        Check the configured kwargs, then load the calculator through mlipx.

        Parameters
        ----------
        **kwargs
            Any keyword arguments to pass to `get_calculator`, including the dtype
            set from the requested precision.

        Returns
        -------
        Calculator
            Loaded ASE Calculator.
        """
        target = import_calculator(self.module, self.class_name)
        check_kwargs(
            self.kwargs or {},
            target,
            self.class_name,
            reserved=("default_dtype", "dtype"),
        )

        # Drop the precision dtype if unsupported, rather than letting ASE ignore it
        accepted = accepted_kwargs(target, self.class_name)
        for dtype_arg in ("default_dtype", "dtype"):
            if dtype_arg in kwargs and dtype_arg not in accepted:
                warn(
                    f"{self.class_name} does not take '{dtype_arg}', so the requested "
                    "precision has no effect",
                    stacklevel=2,
                )
                del kwargs[dtype_arg]

        # mlipx passes `device` to every calculator, so drop it the same way
        model = self
        if self.device is not None and "device" not in accepted:
            warn(
                f"{self.class_name} does not take 'device', so it is ignored. Set "
                "'device: null' for this model to make that explicit",
                stacklevel=2,
            )
            model = dataclasses.replace(self, device=None)

        return MlipxGenericASECalc.get_calculator(model, **kwargs)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Calculator.
        """
        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["default_dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["default_dtype"] = self.default_dtype

        return self.load_calculator(**kwargs)


# https://github.com/isayevlab/aimnetcentral
@dataclasses.dataclass(kw_only=True)
class AIMNetCalc(SumCalc):
    """Dataclass for AIMNet2 calculators."""

    device: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the AIMNet2 ASE calculator.

        Each structure's charge (and multiplicity, for NSE models) is read from
        ``atoms.info``, defaulting to a neutral singlet rather than reusing the
        previous structure's values.

        Parameters
        ----------
        precision
            Requested ML-PEG precision. AIMNet2 evaluates its network in float32
            (accumulating energies in float64) and does not support full float64,
            so this argument is ignored.
        **kwargs
            Keyword arguments that override the default ``AIMNet2Calculator``
            options.

        Returns
        -------
        Calculator
            Loaded AIMNet2 ASE calculator.
        """
        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

        class _StatelessChargeAIMNet2ASE(AIMNet2ASE):
            """AIMNet2ASE that does not carry charge/mult between structures."""

            def _update_charge_spin_from_info(self, atoms=None):
                """
                Update the calculator's charge and multiplicity from the ASE Atoms info.

                Parameters
                ----------
                atoms : ase.Atoms, optional
                    The ASE Atoms object to read charge and multiplicity from.
                    If not provided, it will attempt to use the `atoms`
                    attribute of the calculator itself.
                """
                atoms = getattr(self, "atoms", None) if atoms is None else atoms
                if atoms is None:
                    return
                info = atoms.info
                charge = info.get("charge", 0)
                if charge != self.charge:
                    self.charge = charge
                    self._t_charge = None
                if self.base_calc.is_nse:
                    mult = info.get("mult", info.get("spin", 1))
                    if mult != self.mult:
                        self.mult = mult
                        self._t_mult = None

        calc_kwargs = {**self.kwargs, **kwargs}
        check_kwargs(calc_kwargs, AIMNet2Calculator, "AIMNet2ASE", reserved=("device",))

        check_precision(precision)
        device = Device.resolve_auto() if self.device == Device.AUTO else self.device
        base_calc = AIMNet2Calculator(device=device, **calc_kwargs)
        return _StatelessChargeAIMNet2ASE(base_calc)

    @property
    def available(self) -> bool:
        """
        Check whether the AIMNet2 calculator is available.

        Returns
        -------
        bool
            Whether the calculator can be imported.
        """
        try:
            from aimnet.calculators import AIMNet2ASE  # noqa: F401

            return True
        except ImportError:
            return False


@dataclasses.dataclass(kw_only=True)
class DpaCalc(SumCalc):
    """Dataclass for DPA calculators provided by DeePMD-kit."""

    module: str = "deepmd.calculator"
    class_name: str = "DP"
    device: DeviceType | None = None
    default_dtype: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the DeePMD ASE calculator.

        Parameters
        ----------
        precision
            Requested ML-PEG precision. DPA checkpoints are fixed at float32, so
            both ``low`` and ``high`` use the checkpoint's native precision.
        **kwargs
            Keyword arguments that override the registry calculator options.

        Returns
        -------
        Calculator
            Loaded DeePMD ASE calculator.
        """
        check_precision(precision)
        warn_precision_unused(self.class_name, self.default_dtype)

        if self.device is not None:
            warn(
                f"{self.class_name} has no device argument, so 'device: "
                f"{self.device}' is ignored. DeePMD selects the device itself",
                stacklevel=2,
            )

        dp = import_calculator(self.module, self.class_name)
        check_kwargs(self.kwargs, dp, self.class_name)

        calculator_kwargs = {**self.kwargs, **kwargs}
        return dp(**calculator_kwargs)

    @property
    def available(self) -> bool:
        """
        Check whether the DeePMD ASE calculator is available.

        Returns
        -------
        bool
            Whether the calculator can be imported.
        """
        try:
            import_calculator(self.module, self.class_name)

            return True
        except (ImportError, AttributeError):
            return False


@dataclasses.dataclass(kw_only=True)
class NequIPCalc(SumCalc):
    """Dataclass for NequIP framework (NequIP and Allegro) calculators."""

    device: Device | str = Device.AUTO
    compile: bool = True
    default_dtype: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the NequIP ASE calculator.

        Parameters
        ----------
        precision
            Requested ML-PEG precision. Packaged NequIP models fix their model dtype
            (float32 for the OAM models), so both ``low`` and ``high`` use the package's
            native precision.
        **kwargs
            Keyword arguments that override the registry calculator options. ``model``
            is a ``.nequip.zip`` path, or an ``https://`` URL downloaded once. Other
            options are passed to both ``NequIPCalculator`` constructors.

        Returns
        -------
        Calculator
            Loaded NequIP ASE calculator.
        """
        check_precision(precision)
        warn_precision_unused("NequIPCalculator", self.default_dtype)
        from nequip.utils.dtype import torch_default_dtype
        import torch

        calc_kwargs = {
            "chemical_species_to_atom_type_map": True,
            **self.kwargs,
            **kwargs,
        }
        model = calc_kwargs.pop("model")
        if model.startswith("https://"):
            model = _download_nequip_package(model)
        device = Device.resolve_auto() if self.device == Device.AUTO else self.device
        options = json.dumps(calc_kwargs, sort_keys=True)
        # nequip's 1st load makes f64 the default dtype – instead keep caller's dtype:
        with torch_default_dtype(torch.get_default_dtype()):
            calc = _load_nequip_calculator(model, device, self.compile, options)
        # copies share the cached model, but not its results or atoms:
        return copy.copy(calc)


@dataclasses.dataclass(kw_only=True)
class MatterSimCalc(GenericASECalc):
    """Dataclass for MatterSim calculator."""

    @staticmethod
    def _patch_mattersim_setstate() -> None:
        """Rebuild the model from mattersim's own saved model_args on copy()."""
        from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
        from mattersim.forcefield.potential import MatterSimCalculator, Potential
        import torch

        def __setstate__(self, state):  # noqa: N807
            """
            Restore from copy/pickle by rebuilding at the saved architecture.

            Parameters
            ----------
            self
                The ``MatterSimCalculator`` instance being restored.
            state
                State dict produced by ``__getstate__``, containing the saved
                model weights, architecture (``model_args``) and name.
            """
            model_state_dict = state.pop("_model_state_dict")
            model_args = state.pop("_model_args")
            model_name = state.pop("_model_name")
            self.__dict__.update(state)
            model = M3Gnet(device=self.device, **model_args).to(self.device)
            model.load_state_dict(model_state_dict)
            model.eval()
            self.potential = Potential(
                model,
                device=self.device,
                model_name=model_name,
                load_training_state=False,
            )
            if self.dtype == torch.float64:
                self.potential.model.double()

        MatterSimCalculator.__setstate__ = __setstate__

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Calculator.
        """
        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["dtype"] = self.default_dtype

        self._patch_mattersim_setstate()

        return self.load_calculator(**kwargs)


@dataclasses.dataclass(kw_only=True)
class VivaceCalc(SumCalc):
    """Dataclass for Vivace calculator."""

    module: str = "simpoly.vivace.calculator"
    class_name: str = "MLFFCalculator"
    device: DeviceType | None = None
    default_dtype: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Unused precision argument, kept for the common model API.
        **kwargs
            Keyword arguments passed to the Vivace calculator.

        Returns
        -------
        Calculator
            Loaded ASE calculator.
        """
        mlff_calculator = import_calculator(self.module, self.class_name)
        check_kwargs(self.kwargs, mlff_calculator, self.class_name)

        check_precision(precision)
        warn_precision_unused(self.class_name, self.default_dtype)

        kwargs.update(self.kwargs)
        calc = mlff_calculator(**kwargs)

        # Vivace sets dtype from checkpoint metadata inside MLFFCalculator.
        # Leave precision/overwrite_dtype untouched unless SimPoly exposes it.
        device = Device.resolve_auto() if self.device == Device.AUTO else self.device
        if device is not None:
            calc.device = device
            calc.model = calc.model.to(device=device)

        return calc


# https://github.com/orbital-materials/orb-models
@dataclasses.dataclass(kw_only=True)
class OrbCalc(SumCalc):
    """Dataclass for Orb calculator."""

    name: str
    module: str = "orb_models.forcefield.inference.calculator"
    class_name: str = "ORBCalculator"
    #: Module providing the pretrained checkpoint loaders selected by `name`.
    pretrained_module: str = "orb_models.forcefield.pretrained"
    device: DeviceType | None = None
    default_dtype: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded ASE Orb Calculator.
        """
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
        import os

        check_precision(precision)
        os.environ["TORCH_DISABLE_MODULE_HIERARCHY_TRACKING"] = "1"

        orb_calculator = import_calculator(self.module, self.class_name)
        method = import_calculator(self.pretrained_module, self.name)

        precision_map = {"low": "float32-high", "high": "float64"}
        dtype = precision_map[precision]

        if self.default_dtype is not None:
            dtype = self.default_dtype

        model_kwargs, calc_kwargs = split_kwargs(
            self.kwargs,
            (method, orb_calculator),
            reserved=("device", "precision", "model", "atoms_adapter"),
        )

        device = Device.resolve_auto() if self.device == Device.AUTO else self.device

        if device is None:
            orbff, atoms_adapter = method(precision=dtype, **model_kwargs)
            return orb_calculator(orbff, atoms_adapter=atoms_adapter, **calc_kwargs)

        orbff, atoms_adapter = method(device=device, precision=dtype, **model_kwargs)
        return orb_calculator(
            orbff, atoms_adapter=atoms_adapter, device=device, **calc_kwargs
        )

    @property
    def available(self) -> bool:
        """
        Check whether the calculator module is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            # Same imports as `get_calculator`, so the check cannot drift from it
            import_calculator(self.module, self.class_name)
            import_calculator(self.pretrained_module, self.name)

            return True
        except (ImportError, AttributeError):
            return False


@dataclasses.dataclass(kw_only=True)
class FairChemCalc(SumCalc):
    """Dataclass for fairchem (UMA) calculator."""

    model_name: str
    task_name: str
    module: str = "fairchem.core"
    class_name: str = "FAIRChemCalculator"
    device: DeviceType = "cpu"
    default_dtype: str | None = None
    # fairchem inference preset. "batch" matches the fairchem<2.25 default.
    inference_settings: str = "batch"
    overrides: dict = dataclasses.field(default_factory=dict)
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any additional keyword arguments.

        Returns
        -------
        Calculator
            Loaded ASE fairchem Calculator.
        """
        from fairchem.core.units.mlip_unit.api.inference import (
            guess_inference_settings,
        )

        fairchem_calculator = import_calculator(self.module, self.class_name)
        # Checkpoints are loaded from the same module as the calculator.
        pretrained_mlip = import_calculator(self.module, "pretrained_mlip")

        # fairchem defaults to float32; map the requested precision to the base
        # dtype so precision="high" runs in float64. A configured default_dtype
        # overrides this.
        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        dtype = self.default_dtype or precision_map[precision]
        settings = guess_inference_settings(self.inference_settings)
        # UMA Nitro only runs in float32, so use the standard path otherwise,
        # rather than letting fairchem fall back to its TF32 multigpu preset.
        if settings.nitro and dtype != "float32":
            settings = guess_inference_settings("batch")
        inference_settings = dataclasses.replace(settings, base_precision_dtype=dtype)

        predictor_kwargs, calc_kwargs = split_kwargs(
            self.kwargs,
            (pretrained_mlip.get_predict_unit, fairchem_calculator),
            reserved=(
                "model_name",
                "device",
                "overrides",
                "inference_settings",
                "task_name",
                "predict_unit",
            ),
        )

        device = Device.resolve_auto() if self.device == Device.AUTO else self.device

        predictor = pretrained_mlip.get_predict_unit(
            self.model_name,
            device=device,
            overrides=self.overrides,
            inference_settings=inference_settings,
            **predictor_kwargs,
        )
        return fairchem_calculator(predictor, task_name=self.task_name, **calc_kwargs)

    @property
    def available(self) -> bool:
        """
        Check whether the calculator module is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            pretrained_mlip = import_calculator(self.module, "pretrained_mlip")

            return self.model_name in pretrained_mlip._MODEL_CKPTS.checkpoints
        except Exception:
            return False


@dataclasses.dataclass(kw_only=True)
class MockCalc(SumCalc):
    """Dataclass for mock calculator."""

    model_name: str = "mock"
    trained_on_dispersion: bool = True

    def get_calculator(self, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        **kwargs
            Any additional keyword arguments passed to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded mock ASE Calculator.
        """
        from ml_peg.models.mock import MockCalculator

        return MockCalculator()


@dataclasses.dataclass(kw_only=True)
class UPETCalc(GenericASECalc):
    """Dataclass for upet (PET-MAD / PET-OAM) calculator."""

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Any keyword arguments to pass to `get_calculator`.

        Returns
        -------
        Calculator
            Loaded upet ASE calculator.
        """
        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["dtype"] = self.default_dtype

        _patch_metatomic_nvalchemi_max_neighbors()
        return self.load_calculator(**kwargs)


@dataclasses.dataclass(kw_only=True)
class SevenNetCalc(SumCalc):
    """Dataclass for SevenNet calculator."""

    module: str = "sevenn.calculator"
    class_name: str = "SevenNetCalculator"
    device: DeviceType | None = None
    default_dtype: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Currently unused. Once supported, this will correspond to the level of
            precision to evaluate the model.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        Calculator
            Loaded SevenNet ASE calculator.
        """
        sevennet_calculator = import_calculator(self.module, self.class_name)
        check_kwargs(self.kwargs, sevennet_calculator, self.class_name)

        check_precision(precision)
        warn_precision_unused(self.class_name, self.default_dtype)
        device = Device.resolve_auto() if self.device == Device.AUTO else self.device

        if device is None:
            return sevennet_calculator(**self.kwargs)

        return sevennet_calculator(device=device, **self.kwargs)


# https://github.com/xvzemin/tace
@dataclasses.dataclass(kw_only=True)
class TaceCalc(SumCalc):
    """Dataclass for TACE calculators."""

    module: str = "tace.interface.ase"
    class_name: str = "TACEAseCalc"
    device: DeviceType | None = None
    default_dtype: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Keyword arguments that override the registry calculator options.

        Returns
        -------
        Calculator
            Loaded TACE ASE calculator.
        """
        tace_calculator = import_calculator(self.module, self.class_name)
        check_kwargs(
            self.kwargs, tace_calculator, self.class_name, reserved=("dtype", "device")
        )

        check_precision(precision)
        precision_map = {"low": "float32", "high": "float64"}
        dtype = precision_map[precision]

        if self.default_dtype is not None:
            dtype = self.default_dtype

        device = Device.resolve_auto() if self.device == Device.AUTO else self.device

        # TACE resolves foundation names and local checkpoint paths directly.
        calc_kwargs = {**self.kwargs, **kwargs}
        return tace_calculator(dtype=dtype, device=device, **calc_kwargs)

    @property
    def available(self) -> bool:
        """
        Check whether the calculator module is available.

        Returns
        -------
        bool
            Whether the calculator can be loaded.
        """
        try:
            # Same import as `get_calculator`, so the check cannot drift from it
            import_calculator(self.module, self.class_name)

            return True
        except (ImportError, AttributeError):
            return False


@dataclasses.dataclass(kw_only=True)
class GraceCalc(GenericASECalc):
    """Dataclass for GRACE calculator."""

    device: DeviceType | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        precision
            Level of precision to evaluate the model.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        Calculator
            Loaded GRACE ASE calculator.
        """
        from tensorpotential.calculator.foundation_models import MODELS_NAME_LIST

        check_precision(precision)
        precision_map = {"low": "", "high": "-fp64"}
        suffix = precision_map[precision]

        if self.default_dtype is not None:
            suffix = self.default_dtype

        configured_model = self.kwargs.get("model")
        if configured_model is None:
            raise ValueError(f"{self.class_name} models must set 'model' in kwargs")

        model_name = f"{configured_model}{suffix}"
        if model_name not in MODELS_NAME_LIST:
            warn(
                f"No '{model_name}' checkpoint is published, so '{configured_model}' "
                "is used at its own precision instead",
                stacklevel=2,
            )
            model_name = configured_model

        # Set on a copy, as `self.kwargs` is shared with the model registry
        model = dataclasses.replace(self, kwargs={**self.kwargs, "model": model_name})

        return model.load_calculator(**kwargs)


@dataclasses.dataclass(kw_only=True)
class TorchMDNetCalc(SumCalc):
    """Dataclass for TorchMD-Net calculators, such as AceFF."""

    repo_id: str
    filename: str
    revision: str | None = None
    module: str = "torchmdnet.calculators"
    class_name: str = "TMDNETCalculator"
    device: Device | str = Device.AUTO
    default_dtype: str | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, *, precision: Precision, **kwargs) -> Calculator:
        """
        Prepare and load the TorchMD-Net ASE calculator.

        Parameters
        ----------
        precision
            Requested ML-PEG precision. ``TMDNETCalculator`` always builds its input
            tensors in float32, so the model only runs in float32 (``dtype=float64``
            fails with a dtype mismatch). Both ``low`` and ``high`` use float32, and
            ``overwrite_dtype`` has no effect.
        **kwargs
            Keyword arguments that override the registry calculator options.

        Returns
        -------
        Calculator
            Loaded TorchMD-Net ASE calculator.
        """
        check_precision(precision)
        if self.default_dtype is not None:
            warn(
                f"{self.class_name} builds float32 inputs, so overwrite_dtype "
                f"'{self.default_dtype}' has no effect",
                stacklevel=2,
            )

        tmdnet_calculator = import_calculator(self.module, self.class_name)
        calculator_kwargs = {**self.kwargs, **kwargs}
        check_kwargs(
            calculator_kwargs,
            tmdnet_calculator,
            self.class_name,
            reserved=("model_file", "device", "dtype"),
        )

        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            repo_id=self.repo_id, filename=self.filename, revision=self.revision
        )
        device = Device.resolve_auto() if self.device == Device.AUTO else self.device
        return tmdnet_calculator(
            model_file=model_file, device=device, **calculator_kwargs
        )

    @property
    def available(self) -> bool:
        """
        Check whether the TorchMD-Net ASE calculator is available.

        Returns
        -------
        bool
            Whether the calculator can be imported.
        """
        try:
            import_calculator(self.module, self.class_name)

            return True
        except (ImportError, AttributeError):
            return False
