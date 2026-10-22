"""Tests for the AceFF model and its TorchMDNet calculator wrapper."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ase.build import molecule
import numpy as np
import pytest
import yaml

from ml_peg.models.get_models import load_models
from ml_peg.models.models import TorchMDNetCalc

ACEFF_MODEL = "aceff-2p0"
ACEFF_REPO_ID = "Acellera/AceFF-2.0"
ACEFF_FILENAME = "aceff_v2.0.ckpt"
ACEFF_REVISION = "67ad6a3b06ee9b44cfbac3bc32117dd6276e9122"

# Reference AceFF-2.0 (pinned revision) predictions for ``molecule("H2O")``, with
# TorchMD-Net's default atomic reference energy removal. Units: eV and eV/Å.
WATER_REFERENCE = {
    0: (
        -0.98872077,
        [
            [0.0, 0.0, -0.829859],
            [0.0, -0.406309, 0.41493],
            [0.0, 0.40631, 0.414929],
        ],
    ),
    -1: (
        0.14240241,
        [
            [0.0, 0.0, 0.358114],
            [0.0, 0.450995, -0.179057],
            [0.0, -0.450995, -0.179057],
        ],
    ),
}


@pytest.fixture(scope="module")
def aceff() -> TorchMDNetCalc:
    """
    Load the registered AceFF model.

    Returns
    -------
    TorchMDNetCalc
        Registered AceFF model wrapper.
    """
    pytest.importorskip("torchmdnet")
    return load_models((ACEFF_MODEL,))[ACEFF_MODEL]


@pytest.fixture(scope="module")
def calculator(aceff):
    """
    Build the AceFF calculator once, downloading the pinned checkpoint.

    Parameters
    ----------
    aceff
        Registered AceFF model wrapper.

    Returns
    -------
    Calculator
        AceFF TorchMD-Net calculator.
    """
    return aceff.get_calculator(precision="high")


@pytest.fixture
def water(calculator):
    """
    Build a water molecule without charge information, using the AceFF calculator.

    Parameters
    ----------
    calculator
        AceFF TorchMD-Net calculator.

    Returns
    -------
    Atoms
        Water molecule with the calculator attached and cached results reset.
    """
    calculator.reset()
    atoms = molecule("H2O")
    atoms.calc = calculator
    return atoms


def test_aceff_registry_metadata():
    """The AceFF entry records its wrapper, pinned checkpoint, and D3 behaviour."""
    registry_path = Path(__file__).parents[1] / "ml_peg" / "models" / "models.yml"
    config = yaml.safe_load(registry_path.read_text(encoding="utf8"))[ACEFF_MODEL]

    assert config["class_name"] == "TMDNETCalculator"
    assert "overwrite_dtype" not in config
    assert config["trained_on_dispersion"]
    assert config["level_of_theory"] == "ωB97M-V/def2-TZVPPD"
    assert config["kwargs"] == {
        "repo_id": ACEFF_REPO_ID,
        "filename": ACEFF_FILENAME,
        "revision": ACEFF_REVISION,
    }


def test_aceff_uses_dedicated_wrapper(aceff):
    """The registered AceFF model routes through the TorchMD-Net wrapper."""
    assert isinstance(aceff, TorchMDNetCalc)
    assert aceff.available
    assert aceff.trained_on_dispersion


def test_aceff_downloads_pinned_checkpoint(calculator):
    """The checkpoint is downloaded at the pinned revision."""
    model_file = Path(calculator.model_file)

    assert model_file.name == ACEFF_FILENAME
    assert model_file.parent.name == ACEFF_REVISION


@pytest.mark.parametrize("charge", WATER_REFERENCE)
def test_aceff_reference_energy_and_forces(water, charge):
    """The pinned checkpoint reproduces reference energies and forces."""
    energy, forces = WATER_REFERENCE[charge]
    water.info["charge"] = charge

    assert water.get_potential_energy() == pytest.approx(energy, abs=1e-4)
    np.testing.assert_allclose(water.get_forces(), forces, atol=1e-4)


@pytest.mark.parametrize("charge", WATER_REFERENCE)
def test_aceff_matches_torchmdnet_calculator(calculator, water, charge):
    """The wrapper gives the same results as TorchMD-Net's own calculator."""
    from torchmdnet.calculators import TMDNETCalculator

    water.info["charge"] = charge
    reference = water.copy()
    reference.calc = TMDNETCalculator(calculator.model_file, device=calculator.device)

    assert water.get_potential_energy() == pytest.approx(
        reference.get_potential_energy()
    )
    np.testing.assert_allclose(water.get_forces(), reference.get_forces())


def test_aceff_forces_match_energy_gradient(water):
    """Forces agree with central finite differences of the energy."""
    forces = water.get_forces()
    step = 1e-3
    numerical = np.zeros_like(forces)
    for index in np.ndindex(forces.shape):
        displaced = []
        for sign in (1, -1):
            atoms = water.copy()
            atoms.calc = water.calc
            atoms.positions[index] += sign * step
            displaced.append(atoms.get_potential_energy())
        numerical[index] = -(displaced[0] - displaced[1]) / (2 * step)

    np.testing.assert_allclose(forces, numerical, atol=5e-3)


@pytest.mark.parametrize(
    "kwargs", [{}, {"repo_id": ACEFF_REPO_ID}, {"filename": ACEFF_FILENAME}]
)
def test_aceff_requires_checkpoint(tmp_path, kwargs):
    """A registry entry without a complete Hub checkpoint fails to load."""
    registry = {
        "broken": {
            "module": "torchmdnet.calculators",
            "class_name": "TMDNETCalculator",
            "trained_on_dispersion": True,
            "kwargs": kwargs,
        }
    }
    models_file = tmp_path / "models.yml"
    models_file.write_text(yaml.safe_dump(registry), encoding="utf8")

    with pytest.raises(ValueError, match="'broken' must set"):
        load_models(("broken",), filepath=models_file, run_mock=False, mock_only=False)


def test_aceff_rejects_unknown_kwargs():
    """Misspelled keyword arguments are rejected rather than silently ignored."""
    pytest.importorskip("torchmdnet")
    model = TorchMDNetCalc(
        repo_id=ACEFF_REPO_ID, filename=ACEFF_FILENAME, kwargs={"compil": True}
    )

    with pytest.raises(ValueError, match="compil would be ignored"):
        model.get_calculator(precision="low")


@pytest.mark.parametrize("reserved", ["device", "dtype", "model_file"])
def test_aceff_rejects_reserved_kwargs(reserved):
    """Arguments ML-PEG sets itself cannot also be set through kwargs."""
    pytest.importorskip("torchmdnet")
    model = TorchMDNetCalc(
        repo_id=ACEFF_REPO_ID, filename=ACEFF_FILENAME, kwargs={reserved: "x"}
    )

    with pytest.raises(ValueError, match=f"{reserved} is set by ML-PEG"):
        model.get_calculator(precision="low")


def test_aceff_warns_overwrite_dtype_unused(aceff):
    """``overwrite_dtype`` cannot change the float32 inputs, so it warns."""
    model = dataclasses.replace(aceff, default_dtype="float64")

    with pytest.warns(UserWarning, match="overwrite_dtype 'float64' has no effect"):
        model.get_calculator(precision="high")


def test_aceff_defaults_to_neutral_charge(water):
    """Structures without ``info["charge"]`` are evaluated as neutral."""
    default_energy = water.get_potential_energy()

    neutral = water.copy()
    neutral.info["charge"] = 0
    neutral.calc = water.calc

    assert water.info["charge"] == 0
    assert np.isfinite(default_energy)
    assert neutral.get_potential_energy() == pytest.approx(default_energy)


def test_aceff_uses_explicit_charge(water):
    """An explicit total charge is kept and changes the predicted energy."""
    neutral_energy = water.get_potential_energy()
    water.info["charge"] = -1

    charged_energy = water.get_potential_energy()

    assert water.info["charge"] == -1
    assert np.isfinite(charged_energy)
    assert charged_energy != pytest.approx(neutral_energy)


def test_aceff_caches_results_with_default_charge(water):
    """Setting the default charge does not invalidate cached results."""
    calc = water.calc
    evals = calc.evals

    water.get_potential_energy()
    water.get_potential_energy()
    water.get_forces()

    assert calc.evals == evals + 1


def test_aceff_recalculates_when_structure_or_charge_changes(water):
    """Changing the charge or positions triggers a new evaluation."""
    calc = water.calc
    evals = calc.evals

    water.get_potential_energy()
    water.info["charge"] = 1
    water.get_potential_energy()
    water.positions[0, 0] += 0.01
    water.get_potential_energy()

    assert calc.evals == evals + 3


@pytest.mark.parametrize("precision", ["low", "high"])
def test_aceff_runs_in_float32(aceff, precision):
    """Both ML-PEG precisions run in float32, as ``TMDNETCalculator`` requires."""
    torch = pytest.importorskip("torch")
    calc = aceff.get_calculator(precision=precision)
    atoms = molecule("H2O")
    atoms.calc = calc

    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    parameter_dtypes = {
        parameter.dtype
        for parameter in calc.model.parameters()
        if parameter.is_floating_point()
    }
    assert parameter_dtypes == {torch.float32}


def test_aceff_rejects_unknown_precision():
    """The wrapper rejects unknown ML-PEG precision choices."""
    model = TorchMDNetCalc(repo_id=ACEFF_REPO_ID, filename=ACEFF_FILENAME)

    with pytest.raises(ValueError, match="Invalid precision"):
        model.get_calculator(precision="medium")
