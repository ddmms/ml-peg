"""Run calculation for aqueous FeCl oxidation states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.geometry.rdf import get_rdf
from ase.io import read
from janus_core.calculations.md import NPT
import numpy as np
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
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
            file_prefix=OUT_PATH / f"{salt}_{model_name}",
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

    rmax = 6.0
    nbins = 200
    for salt in IRON_SALTS:
        rdf_list = []
        md_path = OUT_PATH / f"{salt}_{model_name}-traj.extxyz"
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
        rdf_path = OUT_PATH / f"O-Fe_{salt}_{model_name}.rdf"
        np.savetxt(
            rdf_path,
            rdf_data,
            header="r g(r)",
        )
