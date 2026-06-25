"""Run calculations for oxygen adatom diffusion barriers on 2D TMDs."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.neb import NEB
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

COMPOUNDS = ["MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2"]


@pytest.fixture(scope="module")
def relaxed_structs() -> dict[str, Atoms]:
    """
    Run geometry optimisation on all structures.

    Returns
    -------
    dict[str, Atoms]
        Relaxed structures indexed by structure name and model name.
    """
    relaxed_structs = {}

    structure_dir = (
        download_s3_data(
            key="inputs/nebs/O_diffusion_2D_TMDs/O_diffusion_2D_TMDs.zip",
            filename="O_diffusion_2D_TMDs.zip",
        )
        / "O_diffusion_2D_TMDs"
    )

    for model_name, calc in MODELS.items():
        for compound in COMPOUNDS:
            for state in ["initial", "end"]:
                struct_name = f"{compound}_{state}.xyz"
                struct = read(structure_dir / struct_name)
                struct.calc = calc.get_calculator(precision="high")
                struct.info.setdefault("charge", 0)
                struct.info.setdefault("spin", 1)

                geomopt = GeomOpt(
                    struct=struct,
                    write_results=True,
                    file_prefix=OUT_PATH / model_name / struct_name,
                    filter_class=None,
                )
                geomopt.run()
                relaxed_structs[f"{struct_name}-{model_name}"] = geomopt.struct
    return relaxed_structs


@pytest.mark.slow
@pytest.mark.parametrize("compound", COMPOUNDS)
@pytest.mark.parametrize("model_name", MODELS)
def test_o_diffusion(
    relaxed_structs: dict[str, Atoms], model_name: str, compound: str
) -> None:
    """
    Run calculations required for oxygen adatom diffusion on a TMD compound.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    compound
        Name of compound to use.
    """
    try:
        NEB(
            init_struct=relaxed_structs[f"{compound}_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"{compound}_end.xyz-{model_name}"],
            n_images=11,
            interpolator="pymatgen",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / model_name / f"O_diffusion_{compound}",
        ).run()
    except Exception:
        NEB(
            init_struct=relaxed_structs[f"{compound}_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"{compound}_end.xyz-{model_name}"],
            n_images=11,
            interpolator="ase",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / model_name / f"O_diffusion_{compound}",
        ).run()
