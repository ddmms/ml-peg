"""End-to-end precision checks for registered PET calculators.

Run with::

    uv run --extra upet pytest --run-slow -v tests/model_precision/test_pet.py

The model list is read from the selected models YAML by filtering active PET
entries. PET returns forces in float64 containers at both precisions, so the
test checks whether the numerical results lie on the float32 grid.
"""

from __future__ import annotations

from ase.build import bulk
import numpy as np
import pytest

from ml_peg.models.get_models import get_model_names, load_model_configs, load_models

PET_MODULE = "upet.calculator"
MODEL_CONFIGS, _ = load_model_configs(get_model_names())
PET_MODEL_NAMES = tuple(
    name for name, config in MODEL_CONFIGS.items() if config.get("module") == PET_MODULE
)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", PET_MODEL_NAMES)
@pytest.mark.parametrize(
    ("precision", "expected_on_float32_grid"),
    [("low", True), ("high", False)],
)
def test_registered_pet_model_output_precision(
    model_name: str,
    precision: str,
    expected_on_float32_grid: bool,
) -> None:
    """Each registered PET model evaluates forces at the requested precision."""
    pytest.importorskip("upet")
    model = load_models((model_name,))[model_name]

    atoms = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    atoms.positions[0] += [0.071, -0.043, 0.029]
    atoms.info.update(charge=0, spin=0)
    atoms.calc = model.get_calculator(precision=precision)
    forces = atoms.get_forces()

    forces_on_float32_grid = np.array_equal(
        forces,
        forces.astype(np.float32).astype(np.float64),
    )
    assert forces.dtype == np.dtype("float64")
    assert forces_on_float32_grid is expected_on_float32_grid
    assert np.isfinite(forces).all()
