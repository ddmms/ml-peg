"""Run calculations for Surface reaction tests."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.neb import NEB
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

structs = DATA_PATH.glob("*")

REACTIONS = [
    "desorption_ood_87_9841_0_111-1",
    "dissociation_ood_268_6292_46_211-5",
    "transfer_id_601_1482_1_211-5",
    #'desorption_ood_249_10303_1_111-1',
    # 'desorption_id_240_3075_0_111-1',
    # 'desorption_id_182_8153_14_011-5',
    # 'desorption_id_181_11171_3_100-1',
    # 'desorption_id_150_10251_16_211-2',
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
    # 'transfer_ood_578_10006_4_200-1',
    # 'transfer_ood_594_10582_21_100-2',
    # 'transfer_id_610_1068_2_100-2',
    # 'transfer_id_641_8304_21_011-2',
    # 'transfer_id_295_6103_4_211-0',
    # 'transfer_id_171_9735_4_222-0',
    # 'transfer_ood_567_3215_0_111-0',
    # 'transfer_ood_519_6938_8_200-5',
    # 'transfer_id_306_9788_21_111-0',
]


@pytest.fixture(scope="module")
def relaxed_structs() -> dict[str, list[Atoms]]:
    """
    Relax initial and final from DFT trajectory.

    Returns
    -------
    dict[str, list[Atoms]]
        Relaxed structures indexed by structure name and model name.
    """
    relaxed_structs = {}

    for reaction in REACTIONS:
        for model_name, calc in MODELS.items():
            initial = read(DATA_PATH / reaction, "0")
            final = read(DATA_PATH / reaction, "-1")
            fix_indices = [
                i for i, tag in enumerate(initial.arrays["tags"]) if tag == 0
            ]
            for struct in [initial, final]:
                struct.calc = calc.get_calculator()
                struct.constraints = [FixAtoms(indices=fix_indices)]

                geomopt = GeomOpt(
                    struct=struct,
                    write_results=True,
                    file_prefix=OUT_PATH / f"{reaction}-{model_name}",
                    filter_class=None,
                )
                geomopt.run()
            relaxed_structs[f"{reaction}-{model_name}"] = [initial, final]
    return relaxed_structs


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_surface_reaction(
    relaxed_structs: dict[str, list[Atoms]], model_name: str
) -> None:
    """
    Run calculations required for surface reactions.

    Parameters
    ----------
    relaxed_structs
        Relax and make interpolation, indexed by reaction name and model name.
    model_name
        Name of model to use.
    """
    for reaction in REACTIONS:
        neb = NEB(
            init_struct=relaxed_structs[f"{reaction}-{model_name}"][0],
            final_struct=relaxed_structs[f"{reaction}-{model_name}"][-1],
            n_images=10,
            interpolator="pymatgen",
            minimize=True,
            plot_band=True,
            write_band=True,
            file_prefix=OUT_PATH / f"{reaction}-{model_name}",
        )
        neb.interpolate()
        neb.interpolator = None
        neb.run()
