"""Run calculation for aqueous FeCl oxidation states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.geometry.rdf import get_rdf
from ase.io import read
from janus_core.calculations.md import NPT
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH =  download_s3_data(
    filename="oxidation_states.zip",
    key="inputs/physicality/oxidation_states/oxidation_states.zip",
) / Path("oxidation_states")
print(DATA_PATH)
OUT_PATH = Path(__file__).parent / "outputs"

IRON_SALTS = ["Fe2Cl", "Fe3Cl"]


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_iron_oxidation_state_md(mlip: tuple[str, Any]) -> None:
    """
    Run MLMD for aqueous FeCl oxidation states tests.

    Parameters
    ----------
    mlip
        Name of model use and model.
    """
    model_name, model = mlip
    model.default_dtype = "float32"
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for salt in IRON_SALTS:
        struct_path = DATA_PATH / f"{salt}_start.xyz"
        struct = read(struct_path, "0")
        struct.calc = calc

        npt = NPT(
            struct=struct,
            steps=40000,
            timestep=0.5,
            stats_every=50,
            traj_every=100,
            traj_append=True,
            thermostat_time=50,
            barostat_time=None,
            file_prefix=out_dir / f"{salt}_{model_name}",
        )
        npt.run()


@pytest.mark.parametrize("mlip", MODELS.items())
def test_iron_oxygen_rdfs(mlip: tuple[str, Any]) -> None:
    """
    Calculate Fe-O RDFs from NVT MLMD for oxidation states tests.

    Parameters
    ----------
    mlip
        Name of model used and model.
    """
    model_name, model = mlip
    out_dir = OUT_PATH / model_name

    rmax = 6.0
    nbins = 200
    for salt in IRON_SALTS:
        rdf_list = []
        md_path = out_dir / f"{salt}_{model_name}-traj.extxyz"
        
        if not md_path.exists():
            pytest.skip(
                "MD trajectory data missing, please check for "
                "already existing data on S3 or run"
                "test_iron_oxidation_state_md with -m very_slow"
            )

        traj = read(md_path, ":")
        for atoms in traj:
            rdf, r = get_rdf(
                atoms,
                rmax=rmax,
                nbins=nbins,
                elements=(26, 8),  # Fe (26), O (8)
            )
            rdf_list.append(rdf)
        g_mean = np.mean(rdf_list, axis=0)  # NVT so this is ok

        rdf_data = np.column_stack((r, g_mean))
        rdf_path = out_dir / f"O-Fe_{salt}_{model_name}.rdf"
        np.savetxt(
            rdf_path,
            rdf_data,
            header="r g(r)",
        )
