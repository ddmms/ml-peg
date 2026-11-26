"""Evaluate MLIP forces on small atomic clusters against r2SCAN references."""

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
def test_cluster_forces_r2scan(mlip: tuple[str, Any]) -> None:
    """
    Evaluate native model forces for comparison with r2SCAN references.

    The r2SCAN references contain no dispersion correction, so none is added.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    run_cluster_forces(mlip=mlip, reference="r2scan", out_path=OUT_PATH, add_d3=False)
