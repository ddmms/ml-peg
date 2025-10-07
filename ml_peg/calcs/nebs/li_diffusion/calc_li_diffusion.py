"""Run calculations for lithium diffusion barriers."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.neb import NEB
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.fixture(scope="module")
def relaxed_structs() -> dict[str, Atoms]:
    """
    Run geometry optimisation on all structures.

    Returns
    -------
    dict[str, Atoms]
        Relaxed structures indexed by structure name and model name.
    """
    structs = ("LiFePO4_start_bc.cif", "LiFePO4_end_b.cif", "LiFePO4_end_c.cif")
    relaxed_structs = {}

    for model_name, calc in MODELS.items():
        for struct_name in structs:
            struct = read(DATA_PATH / struct_name)
            struct.calc = calc.get_calculator()

            geomopt = GeomOpt(
                struct=struct,
                write_results=True,
                file_prefix=OUT_PATH / f"{struct_name}-{model_name}",
                filter_class=None,
            )
            geomopt.run()
            relaxed_structs[f"{struct_name}-{model_name}"] = geomopt.struct
    return relaxed_structs


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_li_diffusion_b(relaxed_structs: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for lithium diffusion along path B.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    NEB(
        init_struct=relaxed_structs[f"LiFePO4_start_bc.cif-{model_name}"],
        final_struct=relaxed_structs[f"LiFePO4_end_b.cif-{model_name}"],
        n_images=11,
        interpolator="pymatgen",
        minimize=True,
        plot_band=True,
        write_band=True,
        file_prefix=OUT_PATH / f"li_diffusion_b-{model_name}",
    ).run()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_li_diffusion_c(relaxed_structs: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for lithium diffusion along path C.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    NEB(
        init_struct=relaxed_structs[f"LiFePO4_start_bc.cif-{model_name}"],
        final_struct=relaxed_structs[f"LiFePO4_end_c.cif-{model_name}"],
        n_images=11,
        interpolator="pymatgen",
        minimize=True,
        plot_band=True,
        write_band=True,
        file_prefix=OUT_PATH / f"li_diffusion_c-{model_name}",
    ).run()
