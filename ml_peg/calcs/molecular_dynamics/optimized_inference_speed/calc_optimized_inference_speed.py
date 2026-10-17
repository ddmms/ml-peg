"""
Run optimised inference speed benchmark.

Developers' containers are run on a staged campaign. Staging, building and running
are usually done with ``ml_peg inference-speed``; this wrapper runs a staged
campaign through ``ml_peg calc``. Set ``MLPEG_INFERENCE_CAMPAIGN`` to the campaign
directory, and optionally ``MLPEG_INFERENCE_MODELS`` (comma-separated container
names) and ``MLPEG_INFERENCE_RUNTIME`` (``apptainer`` or ``local``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.run import (
    run_campaign,
)


@pytest.mark.very_slow
def test_optimized_inference_speed() -> None:
    """Run every container on every staged structure."""
    campaign = os.environ.get("MLPEG_INFERENCE_CAMPAIGN")
    if not campaign or not (Path(campaign) / "in" / "spec.json").is_file():
        pytest.skip("Set MLPEG_INFERENCE_CAMPAIGN to a staged campaign directory.")
    models = os.environ.get("MLPEG_INFERENCE_MODELS")
    run_campaign(
        Path(campaign),
        models=models.split(",") if models else None,
        runtime=os.environ.get("MLPEG_INFERENCE_RUNTIME", "apptainer"),
    )
