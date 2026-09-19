"""Define classes for all models."""

# ruff: noqa: D101, D102, F401

from __future__ import annotations

import dataclasses
from functools import wraps
from typing import TYPE_CHECKING, Any
from warnings import warn

from mlipx import GenericASECalculator as MlipxGenericASECalc
from mlipx.nodes.generic_ase import Device

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.calculators.mixing import SumCalculator


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

    def get_calculator(self, precision="high", **kwargs) -> Calculator:
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
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["default_dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["default_dtype"] = self.default_dtype

        return MlipxGenericASECalc.get_calculator(self, **kwargs)


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

    def get_calculator(self, precision="high", **kwargs) -> Calculator:
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
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["dtype"] = self.default_dtype

        self._patch_mattersim_setstate()

        return MlipxGenericASECalc.get_calculator(self, **kwargs)


@dataclasses.dataclass(kw_only=True)
class VivaceCalc(SumCalc):
    """Dataclass for Vivace calculator."""

    device: Device | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, precision="high", **kwargs) -> Calculator:
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
        from simpoly.vivace.calculator import MLFFCalculator

        kwargs.update(self.kwargs)
        calc = MLFFCalculator(**kwargs)

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
    device: Device | None = None
    default_dtype: str = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    @property
    def _use_alchemi_d3(self) -> bool:
        """
        Whether to use Orb's compiled AlchemiDFTD3 correction, not TorchDFTD3.

        Opt in per model with ``use_alchemi_d3: true`` under ``dispersion_kwargs``.
        TorchDFTD3 remains the default for every model.

        Returns
        -------
        bool
            Whether the AlchemiDFTD3 correction is selected.
        """
        return bool(self.dispersion_kwargs.get("use_alchemi_d3", False))

    def add_d3_calculator(self, calcs) -> Calculator | SumCalculator:
        """
        Add dispersion corrections to calculator(s).

        Orb's own D3 correction wraps the model rather than the calculator, so it
        is applied in `get_calculator`. Adding a TorchDFTD3 calculator on top
        would double count dispersion.

        Parameters
        ----------
        calcs
            Calculator, or list of calculators, to add dispersion corrections to.

        Returns
        -------
        SumCalculator | Calculator
            The original calculator(s) when Orb's D3 correction is in use,
            otherwise calculator(s) with a TorchDFTD3 correction added.
        """
        if self._use_alchemi_d3:
            return calcs
        return super().add_d3_calculator(calcs)

    def _add_d3_model(self, orbff):
        """
        Wrap an Orb forcefield with Orb's D3 dispersion correction.

        Parameters
        ----------
        orbff
            Loaded Orb forcefield.

        Returns
        -------
        D3SumModel
            Forcefield summed with the dispersion correction, or the original
            forcefield when Orb's D3 correction is not in use.
        """
        if not self._use_alchemi_d3 or self.trained_on_dispersion:
            return orbff

        from orb_models.forcefield.inference.d3_model import AlchemiDFTD3, D3SumModel

        # `xc` matches the key the TorchDFTD3 path uses, and `functional` matches
        # Orb's own naming. Both are accepted.
        functional = self.dispersion_kwargs.get(
            "functional", self.dispersion_kwargs.get("xc", "PBE")
        )
        d3_kwargs = {
            key: value
            for key, value in self.dispersion_kwargs.items()
            if key in ("cutoff", "k1", "k3", "has_stress", "compile")
        }
        # Compiling the D3 kernel is what makes this correction fast.
        d3_kwargs.setdefault("compile", True)

        return D3SumModel(
            orbff,
            AlchemiDFTD3(
                functional=functional.upper(),
                damping=self.dispersion_kwargs.get("damping", "BJ").upper(),
                **d3_kwargs,
            ),
        )

    def get_calculator(self, precision="high", **kwargs) -> Calculator:
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
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
        import os

        os.environ["TORCH_DISABLE_MODULE_HIERARCHY_TRACKING"] = "1"

        method = getattr(pretrained, self.name)

        precision_map = {"low": "float32-high", "high": "float64"}
        dtype = precision_map[precision]

        if self.default_dtype is not None:
            dtype = self.default_dtype

        if self.device is None:
            orbff, atoms_adapter = method(precision=dtype, **self.kwargs)
            orbff = self._add_d3_model(orbff)
            calc = ORBCalculator(orbff, atoms_adapter=atoms_adapter, **self.kwargs)
        elif self.device == Device.AUTO:
            orbff = method(
                device=Device.resolve_auto(),
                precision=dtype,
                **self.kwargs,
            )
            calc = ORBCalculator(orbff, device=Device.resolve_auto(), **self.kwargs)
        else:
            orbff, atoms_adapter = method(
                device=self.device, precision=dtype, **self.kwargs
            )
            orbff = self._add_d3_model(orbff)
            calc = ORBCalculator(
                orbff, atoms_adapter=atoms_adapter, device=self.device, **self.kwargs
            )

        return calc

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
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            return True
        except ImportError:
            return False


@dataclasses.dataclass(kw_only=True)
class FairChemCalc(SumCalc):
    """Dataclass for fairchem (UMA) calculator."""

    model_name: str
    task_name: str
    device: Device | str = "cpu"
    default_dtype: str | None = None
    overrides: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, precision="high", **kwargs) -> Calculator:
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
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.units.mlip_unit.api.inference import (
            inference_settings_default,
        )

        # fairchem defaults to float32; map the requested precision to the base
        # dtype so precision="high" runs in float64. A configured default_dtype
        # overrides this.
        precision_map = {"low": "float32", "high": "float64"}
        dtype = self.default_dtype or precision_map[precision]
        inference_settings = dataclasses.replace(
            inference_settings_default(), base_precision_dtype=dtype
        )

        predictor = pretrained_mlip.get_predict_unit(
            self.model_name,
            device=self.device,
            overrides=self.overrides,
            inference_settings=inference_settings,
        )
        return FAIRChemCalculator(predictor, task_name=self.task_name)

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
            from fairchem.core import pretrained_mlip

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

    def get_calculator(self, precision="high", **kwargs) -> Calculator:
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
        precision_map = {"low": "float32", "high": "float64"}
        kwargs["dtype"] = precision_map[precision]

        if self.default_dtype is not None:
            kwargs["dtype"] = self.default_dtype

        _patch_metatomic_nvalchemi_max_neighbors()
        return MlipxGenericASECalc.get_calculator(self, **kwargs)


@dataclasses.dataclass(kw_only=True)
class SevenNetCalc(SumCalc):
    """Dataclass for SevenNet calculator."""

    device: Device | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, **kwargs) -> Calculator:
        """
        Prepare and load the calculator.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        Calculator
            Loaded SevenNet ASE calculator.
        """
        from sevenn.sevennet_calculator import SevenNetCalculator

        device = Device.resolve_auto() if self.device == Device.AUTO else self.device
        device_str = device.value if isinstance(device, Device) else (device or "cpu")
        return SevenNetCalculator(device=device_str, **self.kwargs)


@dataclasses.dataclass(kw_only=True)
class GraceCalc(GenericASECalc):
    """Dataclass for GRACE calculator."""

    device: Device | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, precision="high", **kwargs) -> Calculator:
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

        precision_map = {"low": "", "high": "-fp64"}
        suffix = precision_map[precision]

        if self.default_dtype is not None:
            suffix = self.default_dtype

        model_name = f"{self.kwargs['model']}{suffix}"
        if model_name in MODELS_NAME_LIST:
            self.kwargs["model"] = model_name
        else:
            warn(
                "Unable to find model with requested precision, using default",
                stacklevel=2,
            )

        return MlipxGenericASECalc.get_calculator(self, **kwargs)
