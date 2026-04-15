"""Run calculations for OC20NEB tests."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
from ase.optimize import BFGS
from janus_core.calculations.neb import NEB
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"
S3_KEY = "inputs/nebs/OC20NEB/OC20NEB.zip"
S3_FILENAME = "OC20NEB.zip"

local_files_present = True
if not local_files_present:
    DATA_PATH = download_s3_data(key=S3_KEY, filename=S3_FILENAME) / "OC20NEB"
REACTIONS = [
    str(reaction_file).split("/")[-1].split(".")[0]
    for reaction_file in DATA_PATH.glob("*.xyz")
]


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_oc20neb(model_name: str) -> None:
    """
    Run calculations required for OC20NEB.

    Parameters
    ----------
    model_name
        Name of model to use.
    """
    calc = MODELS[model_name]

    for reaction in REACTIONS:
        print(f"{reaction} {model_name}, start NEB ...")
        dft_traj = read(DATA_PATH / f"{reaction}.xyz", ":")
        initial, final = dft_traj[0], dft_traj[-1]
        for struct in [initial, final]:
            struct.calc = calc.get_calculator()

        neb = NEB(
            init_struct=initial,
            final_struct=final,
            neb_class="DyNEB",
            n_images=8,
            neb_kwargs={"climb": False, "method": "eb", "scale_fmax": 1.0},
            interpolator="ase",
            minimize=True,
            minimize_kwargs={"fmax": 0.05},
            optimizer=BFGS,
            plot_band=False,
            write_band=False,
            file_prefix=OUT_PATH / f"{reaction}-{model_name}",
        )
        neb.run(fmax=0.45, steps=200)

        neb.neb.climb = True
        neb.plot_band = True
        neb.write_band = True
        neb.run(fmax=0.05, steps=300)

        forces = neb.neb.get_forces()
        neb.results["max_force"] = np.sqrt((forces**2).sum(axis=1).max())
        if neb.write_results:
            with open(neb.results_file, "w", encoding="utf8") as out:
                print("#Barrier [eV] | delta E [eV] | Max force [eV/Å] ", file=out)
                print(*neb.results.values(), file=out)
