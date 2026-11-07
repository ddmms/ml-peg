"""Run MAD-BENCH calculations on the MAD-1.6 test subset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

S3_KEY = "inputs/bulk_crystal/mad_bench/mad_bench.zip"
S3_FILENAME = "mad_bench.zip"
STRUCTURES_FILENAME = "mad_bench.xyz"

REFERENCE_FORCES_KEY = "r2scan_forces"
REFERENCE_STRESS_KEY = "r2scan_stress"
REFERENCE_INFO_KEYS = (
    "r2scan_energy",
    "r2scan_atomization_energy",
    "r2scan_free_energy",
    REFERENCE_STRESS_KEY,
)

PRED_FORCES_KEY = "pred_forces"
PRED_ENERGY_KEY = "pred_energy"
PRED_STRESS_KEY = "pred_stress"

# errors that explicitly report an unsupported element, as raised by metatomic models
UNSUPPORTED_ELEMENT_MESSAGES = ("does not support the atomic type",)


def predict_property(
    get_property: Callable[[], Any], shape: tuple[int, ...], description: str
) -> tuple[np.ndarray, str, str]:
    """
    Predict one property and check its shape and finiteness.

    Parameters
    ----------
    get_property
        Function returning the predicted property.
    shape
        Expected shape of the property.
    description
        Property, model and structure, used in warnings.

    Returns
    -------
    tuple[np.ndarray, str, str]
        Prediction, or NaN of the expected shape on failure, a status, and the last line
        of any error message. The status is ``ok``, ``bad_shape``, ``non_finite``,
        ``unsupported_elements`` if the error explicitly reports an unsupported element,
        or ``error`` for any other exception.
    """
    try:
        value = np.asarray(get_property(), dtype=float)
    except Exception as exc:
        lines = str(exc).strip().splitlines() or [type(exc).__name__]
        message = lines[-1].replace('"', "'")[:200]

        if any(text in str(exc) for text in UNSUPPORTED_ELEMENT_MESSAGES):
            status = "unsupported_elements"
        else:
            status = "error"

        warn(f"Error calculating {description}: {message}", stacklevel=2)

        return np.full(shape, np.nan), status, message

    if value.shape != shape:
        status = "bad_shape"
    elif not np.isfinite(value).all():
        status = "non_finite"
    else:
        return value, "ok", ""

    warn(f"Invalid {description}, status {status}", stacklevel=2)

    return np.full(shape, np.nan), status, ""


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_mad_bench(mlip: tuple[str, Any]) -> None:
    """
    Predict forces and energies and stresses where a reference stress exists.

    Parameters
    ----------
    mlip
        Name of model used and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_path = download_s3_data(key=S3_KEY, filename=S3_FILENAME) / "mad_bench"
    structures = read(data_path / STRUCTURES_FILENAME, index=":")

    n_ok = Counter()
    n_stress = 0

    for atoms in tqdm(structures, desc=f"{model_name} MAD-BENCH"):
        has_reference_stress = REFERENCE_STRESS_KEY in atoms.info

        # reference labels are read from the input file during analysis
        del atoms.arrays[REFERENCE_FORCES_KEY]

        for key in REFERENCE_INFO_KEYS:
            atoms.info.pop(key, None)

        # the source has no charge or spin and its reference is non-spin-polarised
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 1)

        atoms.calc = calc
        label = f"{model_name} on {atoms.info['subset']} frame {atoms.info['frame_id']}"

        properties = {
            "forces": (atoms.get_forces, (len(atoms), 3)),
            "energy": (atoms.get_potential_energy, ()),
        }

        # stress is only evaluated for the fully periodic structures with a reference
        if has_reference_stress:
            properties["stress"] = (atoms.get_stress, (6,))
            n_stress += 1

        predictions = {}

        for name, (get_property, shape) in properties.items():
            value, status, message = predict_property(
                get_property, shape, f"{name} for {label}"
            )

            predictions[name] = value
            atoms.info[f"{name}_status"] = status
            n_ok[name] += status == "ok"

            if message:
                atoms.info[f"{name}_error"] = message

        atoms.calc = None
        atoms.arrays[PRED_FORCES_KEY] = predictions["forces"]
        atoms.info[PRED_ENERGY_KEY] = float(predictions["energy"])

        if has_reference_stress:
            atoms.info[PRED_STRESS_KEY] = predictions["stress"]

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / STRUCTURES_FILENAME, structures)

    print(
        f"{model_name} MAD-BENCH valid predictions, "
        f"forces {n_ok['forces']} of {len(structures)}, "
        f"energy {n_ok['energy']} of {len(structures)}, "
        f"stress {n_ok['stress']} of {n_stress}"
    )
