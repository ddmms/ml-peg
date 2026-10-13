"""Trace each model's lattice-trapping S-curves on the brittle_crack cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import write
import pytest

from ml_peg.calcs.fracture.brittle_crack import crack_tools as ct
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"
CASES = ct.load_cases()


def _plane_key(case: dict) -> str:
    """
    Return a key for the cleavage plane of a case (crack plane and front).

    Parameters
    ----------
    case
        Case settings.

    Returns
    -------
    str
        Key such as ``"1-10_110"``.
    """
    return "_".join(
        "".join(str(i) for i in v) for v in (case["crack_surface"], case["crack_front"])
    )


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_brittle_crack(mlip: tuple[str, Any]) -> None:
    """
    Trace a model's S-curves on every case and write frames and constants.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    out = OUT_PATH / model_name
    out.mkdir(parents=True, exist_ok=True)
    symbols = {case["symbol"] for case in CASES.values()}
    if len(symbols) != 1:
        raise NotImplementedError("one material per benchmark run")
    consts = ct.material_constants(calc, symbols.pop())
    gammas, status, by_plane = {}, {}, {}
    for name, case in CASES.items():
        try:
            key = _plane_key(case)
            if key not in by_plane:
                by_plane[key] = ct.cleavage_energy(
                    calc, consts["a0"], case["crack_surface"], case["crack_front"]
                )
            gammas[name] = by_plane[key]
            frames, status[name] = ct.trace_case(
                case, calc, {**consts, "gamma": gammas[name]}
            )
        except Exception as exc:
            warn(f"{model_name} {name}: {exc}", stacklevel=2)
            frames, status[name] = [], f"failed: {exc}"
        write(out / f"{name}.extxyz", frames, format="extxyz")
    (out / "constants.json").write_text(
        json.dumps({**consts, "gamma": gammas, "status": status}, indent=1)
    )
