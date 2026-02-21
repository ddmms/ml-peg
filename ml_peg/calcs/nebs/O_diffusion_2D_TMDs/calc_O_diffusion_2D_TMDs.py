"""Run calculations for oxygen adatom diffusion barriers on 2D TMDs."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.neb import NEB
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

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
def test_o_diffusion_mos2(relaxed_structs: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for oxygen adatom diffusion on MoS2.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    try:
        NEB(
            init_struct=relaxed_structs[f"MoS2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"MoS2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="pymatgen",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_MoS2-{model_name}",
        ).run()
    except Exception:
        NEB(
            init_struct=relaxed_structs[f"MoS2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"MoS2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="ase",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_MoS2-{model_name}",
        ).run()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_o_diffusion_mose2(relaxed_structs: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for oxygen adatom diffusion on MoSe2.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    try:
        NEB(
            init_struct=relaxed_structs[f"MoSe2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"MoSe2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="pymatgen",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_MoSe2-{model_name}",
        ).run()
    except Exception:
        NEB(
            init_struct=relaxed_structs[f"MoSe2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"MoSe2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="ase",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_MoSe2-{model_name}",
        ).run()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_o_diffusion_mote2(relaxed_structs: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for oxygen adatom diffusion on MoTe2.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    try:
        NEB(
            init_struct=relaxed_structs[f"MoTe2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"MoTe2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="pymatgen",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_MoTe2-{model_name}",
        ).run()
    except Exception:
        NEB(
            init_struct=relaxed_structs[f"MoTe2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"MoTe2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="ase",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_MoTe2-{model_name}",
        ).run()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_o_diffusion_ws2(relaxed_structs: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for oxygen adatom diffusion on WS2.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    try:
        NEB(
            init_struct=relaxed_structs[f"WS2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"WS2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="pymatgen",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_WS2-{model_name}",
        ).run()
    except Exception:
        NEB(
            init_struct=relaxed_structs[f"WS2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"WS2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="ase",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_WS2-{model_name}",
        ).run()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_o_diffusion_wse2(relaxed_structs: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for oxygen adatom diffusion on WSe2.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    try:
        NEB(
            init_struct=relaxed_structs[f"WSe2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"WSe2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="pymatgen",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_WSe2-{model_name}",
        ).run()
    except Exception:
        NEB(
            init_struct=relaxed_structs[f"WSe2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"WSe2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="ase",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_WSe2-{model_name}",
        ).run()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_o_diffusion_wte2(relaxed_structs: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for oxygen adatom diffusion on WTe2.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    try:
        NEB(
            init_struct=relaxed_structs[f"WTe2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"WTe2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="pymatgen",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_WTe2-{model_name}",
        ).run()
    except Exception:
        NEB(
            init_struct=relaxed_structs[f"WTe2_initial.xyz-{model_name}"],
            final_struct=relaxed_structs[f"WTe2_end.xyz-{model_name}"],
            n_images=11,
            interpolator="ase",
            minimize=True,
            write_band=True,
            file_prefix=OUT_PATH / f"O_diffusion_WTe2-{model_name}",
        ).run()
