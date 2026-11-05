"""End-to-end precision checks for registered MatterSim calculators.

Run with::

    uv run --extra mattersim pytest --run-slow -v \
        tests/model_precision/test_mattersim.py

The model list is read from the selected models YAML by filtering active
MatterSim entries. The dtype of forces from a real model evaluation is checked
directly.
"""

from __future__ import annotations

from ase.build import bulk
import numpy as np
import pytest

from ml_peg.models.get_models import get_model_names, load_model_configs, load_models

MATTERSIM_MODULE = "mattersim.forcefield"
MODEL_CONFIGS, _ = load_model_configs(get_model_names())
MATTERSIM_MODEL_NAMES = tuple(
    name
    for name, config in MODEL_CONFIGS.items()
    if config.get("module") == MATTERSIM_MODULE
)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MATTERSIM_MODEL_NAMES)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [("low", np.dtype("float32")), ("high", np.dtype("float64"))],
)
def test_registered_mattersim_model_output_precision(
    model_name: str,
    precision: str,
    expected_dtype: np.dtype,
) -> None:
    """Each MatterSim model evaluates forces at the requested precision."""
    pytest.importorskip("mattersim")
    model = load_models((model_name,))[model_name]

    atoms = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    atoms.positions[0] += [0.071, -0.043, 0.029]
    atoms.info.update(charge=0, spin=0)
    atoms.calc = model.get_calculator(precision=precision)
    forces = atoms.get_forces()

    assert forces.dtype == expected_dtype
    assert np.isfinite(forces).all()
