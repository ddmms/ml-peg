"""Run calculations for lithium diffusion barriers."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.neb import NEB
from numpy import nan
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
            struct.calc = calc.get_calculator(precision="high")
            # Set default charge and spin
            struct.info.setdefault("charge", 0)
            struct.info.setdefault("spin", 1)

            geomopt = GeomOpt(
                struct=struct,
                write_results=True,
                file_prefix=OUT_PATH / model_name / struct_name,
                filter_class=None,
                steps=25,
            )
            geomopt.run()
            relaxed_structs[f"{model_name}-{struct_name}"] = geomopt.struct
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
    init_struct = relaxed_structs[f"{model_name}-LiFePO4_start_bc.cif"]
    final_struct = relaxed_structs[f"{model_name}-LiFePO4_end_b.cif"]
    neb = NEB(
        init_struct=init_struct,
        final_struct=final_struct,
        n_images=11,
        interpolator="pymatgen",
        minimize=True,
        plot_band=True,
        write_band=True,
        file_prefix=OUT_PATH / model_name / "li_diffusion_b",
        steps=100,
    )
    # Set default charge and spin for all images
    try:
        neb.interpolate()
        neb.interpolator = None
        for image in neb.images:
            image.info.setdefault("charge", 0)
            image.info.setdefault("spin", 1)
        neb.run()
    except Exception as e:
        print(f"Error running NEB for {model_name} path B: {e}")
        # Write out equivalent data
        out_dir = OUT_PATH / model_name
        write(out_dir / "li_diffusion_b-neb-band.extxyz", [init_struct, final_struct])
        with open(
            out_dir / "li_diffusion_b-neb-results.dat", "w", encoding="utf8"
        ) as out:
            print("#Barrier [eV] | delta E [eV] | Max force [eV/Å] ", file=out)
            print(nan, nan, nan, file=out)


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
    init_struct = relaxed_structs[f"{model_name}-LiFePO4_start_bc.cif"]
    final_struct = relaxed_structs[f"{model_name}-LiFePO4_end_c.cif"]
    neb = NEB(
        init_struct=init_struct,
        final_struct=final_struct,
        n_images=11,
        interpolator="pymatgen",
        minimize=True,
        plot_band=True,
        write_band=True,
        file_prefix=OUT_PATH / model_name / "li_diffusion_c",
        steps=500,
    )
    # Set default charge and spin for all images
    try:
        neb.interpolate()
        neb.interpolator = None
        for image in neb.images:
            image.info.setdefault("charge", 0)
            image.info.setdefault("spin", 1)
        neb.run()
    except Exception as e:
        print(f"Error running NEB for {model_name} path C: {e}")
        # Write out equivalent data
        out_dir = OUT_PATH / model_name
        write(out_dir / "li_diffusion_c-neb-band.extxyz", [init_struct, final_struct])
        with open(
            out_dir / "li_diffusion_c-neb-results.dat", "w", encoding="utf8"
        ) as out:
            print("#Barrier [eV] | delta E [eV] | Max force [eV/Å] ", file=out)
            print(nan, nan, nan, file=out)
