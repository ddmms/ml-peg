"""Define classes for all models."""

# ruff: noqa: D101, D102, F401

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from mlipx import GenericASECalculator as MlipxGenericASECalc
from mlipx.nodes.generic_ase import Device

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase.calculators.mixing import SumCalculator

current_models = None


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
class PetMadCalc(GenericASECalc):
    """Dataclass for PET-MAD calculator."""

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

        return MlipxGenericASECalc.get_calculator(self, **kwargs)


# https://github.com/orbital-materials/orb-models
@dataclasses.dataclass(kw_only=True)
class OrbCalc(SumCalc):
    """Dataclass for Orb calculator."""

    name: str
    device: Device | None = None
    default_dtype: str = None
    kwargs: dict = dataclasses.field(default_factory=dict)

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
class LammpsMaceCalc(SumCalc):
    """Dataclass for MACE via the LAMMPS ``mliap unified`` pair style."""

    model_path: str
    atom_types: dict | None = None
    kokkos: bool = False
    dispersion: bool = False

    def get_calculator(self, **kwargs) -> Calculator:
        """
        Build a LAMMPSlib calculator using the ``mliap unified`` pair style.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        Calculator
            LAMMPSlib instance configured for MACE via mliap.
        """
        from ase.calculators.lammpslib import LAMMPSlib

        elements = " ".join(self.atom_types) if self.atom_types else ""
        mliap_style = f"mliap unified {self.model_path} 0"
        mliap_coeff = f"pair_coeff * * mliap {elements}".strip()

        if self.dispersion:
            from ase import units

            damping = self.dispersion_kwargs.get("damping", "bj")
            xc = self.dispersion_kwargs.get("xc", "pbe")
            cutoff = self.dispersion_kwargs.get("cutoff", 40.0 * units.Bohr)
            cn_cutoff = self.dispersion_kwargs.get("cn_cutoff", 30.0 * units.Bohr)
            d3_args = f"dispersion/d3 {damping} {xc} {cutoff} {cn_cutoff}"
            lmpcmds = [
                f"pair_style hybrid/overlay {mliap_style} {d3_args}",
                mliap_coeff,
                f"pair_coeff * * dispersion/d3 {elements}".strip(),
            ]
        else:
            lmpcmds = [f"pair_style {mliap_style}", mliap_coeff]

        extra_cmd_args = (
            ("-k on g 1 -sf kk -pk kokkos newton on neigh half").split()
            if self.kokkos
            else ()
        )
        return LAMMPSlib(
            lmpcmds=lmpcmds,
            atom_types=self.atom_types,
            extra_cmd_args=extra_cmd_args,
        )

    def add_d3_calculator(self, calcs) -> Calculator:
        """
        Return the calculator unchanged.

        Parameters
        ----------
        calcs
            LAMMPSlib calculator.

        Returns
        -------
        Calculator
            Unchanged calculator; dispersion is handled within LAMMPS via
            ``dispersion=True``, not via the Python SumCalculator path.
        """
        return calcs


@dataclasses.dataclass(kw_only=True)
class FairChemCalc(SumCalc):
    """Dataclass for fairchem (UMA) calculator."""

    model_name: str
    task_name: str
    device: Device | str = "cpu"
    default_dtype: str = "float32"
    overrides: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self) -> Calculator:
        """
        Prepare and load the calculator.

        Returns
        -------
        Calculator
            Loaded ASE Orb Calculator.
        """
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        # torch.serialization.add_safe_globals([slice])

        predictor = pretrained_mlip.get_predict_unit(
            self.model_name, device=self.device, overrides=self.overrides
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
