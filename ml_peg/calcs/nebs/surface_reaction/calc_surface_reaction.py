"""Run calculations for Surface reaction tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
from ase.mep.dyneb import DyNEB
from ase.optimize import BFGS
from ase.mep.neb import idpp_interpolate, interpolate
from ase.constraints import FixAtoms

import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

structs = DATA_PATH.glob("*")

@pytest.fixture(scope="module")
def make_interpolation() -> dict[str, Atoms]:
    """
    Run geometry optimisation on all structures.

    Returns
    -------
    dict[str, Atoms]
        Relaxed structures indexed by structure name and model name.
    """
    struct_names = [
    #'desorption_ood_249_10303_1_111-1',
    # 'desorption_id_240_3075_0_111-1',
    # 'desorption_id_182_8153_14_011-5',
    # 'desorption_id_181_11171_3_100-1',
    # 'desorption_id_150_10251_16_211-2',
    # 'desorption_ood_87_9841_0_111-1',
    # 'desorption_id_263_4981_0_211-1',
    # 'desorption_id_258_8832_5_200-0',
    # 'desorption_id_15_10898_8_122-6',
    # 'desorption_id_153_9859_14_100-0',
    # 'dissociation_ood_71_9489_25_000-3',
    # 'dissociation_id_281_2644_33_200-4',
    # 'dissociation_ood_393_8094_1_100-6',
    # 'dissociation_id_526_4029_53_200-2',
    # 'dissociation_id_154_1581_1_111-0',
    # 'dissociation_id_628_3261_0_100-1',
    # 'dissociation_id_397_164_30_100-0',
    # 'dissociation_id_181_116_6_111-0',
    # 'dissociation_ood_205_2717_39_111-5',
    # 'dissociation_ood_268_6292_46_211-5',
    # 'transfer_ood_578_10006_4_200-1',
    # 'transfer_ood_594_10582_21_100-2',
    # 'transfer_id_610_1068_2_100-2',
    # 'transfer_id_641_8304_21_011-2',
    # 'transfer_id_295_6103_4_211-0',
    # 'transfer_id_171_9735_4_222-0',
    # 'transfer_ood_567_3215_0_111-0',
    # 'transfer_ood_519_6938_8_200-5',
    # 'transfer_id_306_9788_21_111-0',
     'transfer_id_601_1482_1_211-5']
    interpolations = {}

    for struct_name in struct_names:
        print(f"{struct_name = }")
        for model_name, calc in MODELS.items():
            print(f"{model_name = }")
    
            model = MODELS[model_name]
#            model.device = "cuda"
            calc = calc.get_calculator()
    
            traj = read(DATA_PATH / f"{struct_name}.xyz", ":")
            initial, final = traj[0], traj[-1]
            fix_indices = [i for i, tag in enumerate(initial.arrays["tags"]) if tag == 0]
    
            for config, struct in zip(["initial", "final"], [initial, final]):
                struct.calc = calc
                struct.constraints = [FixAtoms(indices=fix_indices)]
                opt = BFGS(struct)
                conv = opt.run(fmax=0.05, steps=500)
                if not conv:
                    raise 
    
            images = [initial.copy()] + [initial.copy() for _ in range(8)] + [final.copy()]
            interpolate(images, mic=True, apply_constraint=True)
            idpp_interpolate(images, mic=True, traj=None, log=None)
            for struct in images:
                struct.calc = calc

            interpolations[f"{struct_name}-{model_name}"] = images
    return interpolations


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_surface_reaction(make_interpolation: dict[str, Atoms], model_name: str) -> None:
    """
    Run calculations required for lithium diffusion along path B.

    Parameters
    ----------
    relaxed_structs
        Relaxed input structures, indexed by structure name and model name.
    model_name
        Name of model to use.
    """
    images = make_interpolation[f"transfer_id_601_1482_1_211-5-{model_name}"]
    neb = DyNEB(images, k=1.0, climb=False, method="eb",
                allow_shared_calculator=True, scale_fmax=1.0)
    opt = BFGS(neb)
    conv = opt.run(fmax=0.05+0.4, steps=200)
    if conv:
        neb.climb = True
        conv = opt.run(fmax=0.05, steps=300)

    if conv:
        converged = True
    else:
        converged = False

    for at in neb.images:
        at.info["converged"] = converged
        at.info["mlip_energy"] = at.get_potential_energy()
        at.arrays["mlip_forces"] = at.get_forces()
   
    OUT_PATH.mkdir(exist_ok=True, parents=True)    
    write(OUT_PATH / f"{struct_name}_{model_name}.xyz", neb.images)

