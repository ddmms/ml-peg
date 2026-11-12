"""End-to-end precision checks for registered MACE calculators.

Run with::

    uv run --extra mace pytest --run-slow -v tests/model_precision/test_mace.py

The model list is read from ``models.yml`` by filtering active entries whose
``module`` is ``mace.calculators``. Commented-out models are therefore not
collected. This currently excludes ``mace_omol``; its factory supports both
``float32`` and ``float64``, so it will be covered automatically if enabled in
the registry.

MACE preserves its compute precision in the NumPy force array returned through
ASE. Checking ``forces.dtype`` therefore verifies a real model evaluation rather
than only the dtype passed to the calculator constructor.
"""

from __future__ import annotations

from ase.build import bulk
import numpy as np
import pytest

from ml_peg.models.get_models import get_model_names, load_model_configs, load_models

MACE_MODULE = "mace.calculators"
MODEL_CONFIGS, _ = load_model_configs(get_model_names())
MACE_MODEL_NAMES = tuple(
    name
    for name, config in MODEL_CONFIGS.items()
    if config.get("module") == MACE_MODULE
)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MACE_MODEL_NAMES)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [("low", np.dtype("float32")), ("high", np.dtype("float64"))],
)
def test_registered_mace_model_output_precision(
    model_name: str,
    precision: str,
    expected_dtype: np.dtype,
) -> None:
    """Each registered MACE model evaluates forces at the requested precision."""
    pytest.importorskip("mace")
    model = load_models((model_name,))[model_name]

    atoms = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    atoms.positions[0] += [0.071, -0.043, 0.029]
    atoms.info.update(charge=0, spin=0)
    atoms.calc = model.get_calculator(precision=precision)
    forces = atoms.get_forces()

    assert forces.dtype == expected_dtype
    assert np.isfinite(forces).all()
