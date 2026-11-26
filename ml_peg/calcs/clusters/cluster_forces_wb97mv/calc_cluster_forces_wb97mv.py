"""Evaluate MLIP forces on small atomic clusters against ωB97M-V references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ml_peg.calcs.utils.cluster_forces import run_cluster_forces
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.medium
@pytest.mark.parametrize("mlip", MODELS.items())
def test_cluster_forces_wb97mv(mlip: tuple[str, Any]) -> None:
    """
    Evaluate model forces for comparison with ωB97M-V references.

    ωB97M-V includes dispersion through its VV10 term, so a D3 correction is added
    to models that were not trained on dispersion-corrected data.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    run_cluster_forces(mlip=mlip, reference="wb97mv", out_path=OUT_PATH, add_d3=True)
