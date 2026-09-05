"""End-to-end precision checks for the GRACE calculator wrapper.

Run with::

    uv run --extra grace pytest --run-slow -v tests/model_precision/test_grace.py

GRACE returns its final ASE energies and forces in float64 containers for both
model variants, so their NumPy dtype does not reveal the model's compute
precision. After evaluating real forces, these tests inspect the loaded
TensorFlow graph's ``forward_layer_1`` output specification: its internal ``I``
tensor is float32 for the SMAX default model and float64 for its ``-fp64``
variant. The older ``GRACE-2L-OMAT`` export lacks this signature, so its test
only exercises the available default model and does not request fp64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.build import bulk
import numpy as np
import pytest

from ml_peg.models.get_models import load_models

GRACE_MODELS = (
    ("grace-2l-omat", "GRACE-2L-OMAT", False),
    (
        "grace-2l-smax-omat-medium",
        "GRACE-2L-SMAX-OMAT-medium",
        True,
    ),
)
SMAX_MODEL_NAME = "grace-2l-smax-omat-medium"
SMAX_FOUNDATION_MODEL = "GRACE-2L-SMAX-OMAT-medium"


def _calculate(
    model: Any,
    precision: str,
) -> tuple[np.dtype | None, np.ndarray]:
    """Run GRACE and return its internal and numerical outputs."""
    atoms = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    atoms.positions[0] += [0.071, -0.043, 0.029]
    atoms.info.update(charge=0, spin=0)
    calculator = model.get_calculator(precision=precision)
    atoms.calc = calculator
    forces = atoms.get_forces()

    # ASE's result container is float64 for both variants. The first-layer graph
    # tensor retains the precision used by the actual GRACE computation.
    signatures = calculator.models[0].signatures
    compute_dtype = None
    if "forward_layer_1" in signatures:
        output_spec = signatures["forward_layer_1"].structured_outputs["I"]
        compute_dtype = np.dtype(output_spec.dtype.as_numpy_dtype)
    return compute_dtype, forces


@pytest.mark.slow
@pytest.mark.parametrize(("model_name", "foundation_model", "has_fp64"), GRACE_MODELS)
def test_registered_grace_model_precision(
    model_name: str,
    foundation_model: str,
    has_fp64: bool,
) -> None:
    """Each registered GRACE model uses its available precision variants."""
    pytest.importorskip("tensorpotential")
    model = load_models((model_name,))[model_name]
    assert model.kwargs["model"] == foundation_model

    low_dtype, low_forces = _calculate(model, "low")

    if has_fp64:
        assert low_dtype == np.dtype("float32")
        high_dtype, high_forces = _calculate(model, "high")
        assert high_dtype == np.dtype("float64")
        assert not np.array_equal(high_forces, low_forces)
    else:
        # This model has no published -fp64 variant. Only exercise its default
        # fp32 model rather than testing the wrapper's high-precision fallback.
        assert low_dtype is None
        assert np.isfinite(low_forces).all()


@pytest.mark.slow
def test_grace_can_switch_from_high_to_low_precision() -> None:
    """Precision selection is not retained between calculator calls."""
    pytest.importorskip("tensorpotential")
    from ml_peg.models.models import GraceCalc

    model = GraceCalc(
        module="tensorpotential.calculator",
        class_name="grace_fm",
        device="cpu",
        kwargs={"model": SMAX_FOUNDATION_MODEL},
    )

    high_dtype, high_forces = _calculate(model, "high")
    low_dtype, low_forces = _calculate(model, "low")

    assert high_dtype == np.dtype("float64")
    assert low_dtype == np.dtype("float32")
    assert not np.array_equal(high_forces, low_forces)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("overwrite_dtype", "expected_dtype"),
    [("float32", np.dtype("float32")), ("float64", np.dtype("float64"))],
)
def test_grace_respects_overwrite_dtype(
    tmp_path: Path,
    overwrite_dtype: str,
    expected_dtype: np.dtype,
) -> None:
    """The registry precision override controls the model's compute dtype."""
    pytest.importorskip("tensorpotential")
    registry = tmp_path / "models.yml"
    registry.write_text(
        f"""{SMAX_MODEL_NAME}:
  module: tensorpotential.calculator
  class_name: grace_fm
  device: cpu
  trained_on_dispersion: false
  overwrite_dtype: {overwrite_dtype}
  kwargs:
    model: {SMAX_FOUNDATION_MODEL}
""",
        encoding="utf8",
    )
    model = load_models((SMAX_MODEL_NAME,), filepath=registry)[SMAX_MODEL_NAME]

    compute_dtype, _ = _calculate(model, "low")
    assert compute_dtype == expected_dtype
