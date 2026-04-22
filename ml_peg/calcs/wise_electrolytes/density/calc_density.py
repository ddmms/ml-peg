"""
Extract density from pre-computed NPT simulations of LiTFSI/H2O electrolyte.

This benchmark reads pre-computed LAMMPS NPT thermo logs and extracts
equilibrium density for each MLIP model. The simulations use a p16_w42
cell (382 atoms: 16 LiTFSI + 42 H2O) at 298 K, 1 atm.

System: 21 m LiTFSI / H2O electrolyte.
Experimental density: 1.7126 g/cm3
(Gilbert et al., J. Chem. Eng. Data 62, 2056 (2017),
DOI: 10.1021/acs.jced.7b00135).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# --- Configuration -----------------------------------------------------------

# Path to converted data (will be replaced with S3 download for PR)
DATA_ROOT = Path(
    os.environ.get(
        "ML_PEG_WISE_DENSITY_DATA_ROOT",
        "/lus/work/CT9/cin1387/lbrugnoli/prove/ml_peg_benchmark/data/wise_electrolytes/density",
    )
)
OUT_PATH = Path(__file__).parent / "outputs"

MODELS = [
    "matpes-r2scan",
    "mace-mpa-0-medium",
    "mace-omat-0-medium",
    "mace-mp-0b3",
    "mace-mh-1-omat",
    "mace-mh-1-omol",
]

RHO_EXP = 1.7126  # g/cm³  (Gilbert et al., JCED 2017)


# --- Pytest interface (ml-peg convention) ------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_extract_density(model: str) -> None:
    """
    Extract and save density data for one model.

    Parameters
    ----------
    model : str
        Name of the MLIP model to extract density for.
    """
    # Read pre-computed density data
    json_path = DATA_ROOT / model / "density.json"
    if not json_path.exists():
        pytest.skip(f"No density data for {model}")

    with open(json_path) as f:
        result = json.load(f)

    # Save to outputs
    out_dir = OUT_PATH / model
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "density.json", "w") as f:
        json.dump(result, f, indent=2)

    # Basic sanity
    assert result["rho_mean"] > 0, f"Negative density for {model}"
    assert abs(result["rho_error_pct"]) < 50, f"Density error > 50% for {model}"
