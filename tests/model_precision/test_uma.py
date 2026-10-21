"""End-to-end precision checks for registered UMA calculators.

Run with::

    uv run --extra uma pytest --run-slow -v tests/model_precision/test_uma.py \
        --models-file ml_peg/models/models_uma.yml

The model list is read from the selected models YAML by filtering active UMA
entries. The dtype of forces from a real model evaluation is checked directly.
"""

from __future__ import annotations

from ase.build import bulk
import numpy as np
import pytest

from ml_peg.models.get_models import get_model_names, load_model_configs, load_models

UMA_MODULE = "fairchem.core"
MODEL_CONFIGS, _ = load_model_configs(get_model_names())
UMA_MODEL_NAMES = tuple(
    name for name, config in MODEL_CONFIGS.items() if config.get("module") == UMA_MODULE
)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", UMA_MODEL_NAMES)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [("low", np.dtype("float32")), ("high", np.dtype("float64"))],
)
def test_registered_uma_model_output_precision(
    model_name: str,
    precision: str,
    expected_dtype: np.dtype,
) -> None:
    """Each registered UMA model evaluates forces at the requested precision."""
    pytest.importorskip("fairchem")
    model = load_models((model_name,))[model_name]

    atoms = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    atoms.positions[0] += [0.071, -0.043, 0.029]
    atoms.info.update(charge=0, spin=0)
    atoms.calc = model.get_calculator(precision=precision)
    forces = atoms.get_forces()

    assert forces.dtype == expected_dtype
    assert np.isfinite(forces).all()
