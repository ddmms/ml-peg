from __future__ import annotations

from pathlib import Path

from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
import pytest
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from ml_peg.analysis.utils.utils import (
    load_metrics_config,
)
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "split_vacancy" / "outputs"
# OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "split_vacancy"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# same setting as MatBench
# https://github.com/janosh/matbench-discovery/blob/93cc6907ac08b4adaa8391ccc4adf9c015c0dd61/matbench_discovery/structure/symmetry.py#L124
STRUCTURE_MATCHER = StructureMatcher(stol=1.0, scale=False)


def get_rmsd(atoms_1, atoms_2):
    rmsd, max_dist = STRUCTURE_MATCHER.get_rms_dist(
        Structure.from_ase_atoms(atoms_1), Structure.from_ase_atoms(atoms_2)
    )

    return rmsd


def get_hoverdata() -> tuple[list, list, list]:
    # TODO: RMSD could be good hoverdata - think about this. Only needed for formation energy parity plot.
    # TODO: add cell charge?
    mp_ids = []
    formulae = []
    vacant_cations = []

    model_dir = model_dir = CALC_PATH / MODELS[0]
    for material_dir in tqdm(list(model_dir.iterdir())):
        split_dir_name = material_dir.stem.split("-")
        bulk_formula = split_dir_name[0]
        mp_id = f"mp-{split_dir_name[-1]}"

        cation_dirs = [
            p for p in material_dir.iterdir() if p.is_dir()
        ]  # skip pristine supercell.xyz files if present (not used)

        for cation_dir in cation_dirs:
            cation = cation_dir.stem

            mp_ids.append(mp_id)
            formulae.append(bulk_formula)
            vacant_cations.append(cation)

    return mp_ids, formulae, vacant_cations


@pytest.fixture  # cache outputs
def build_results() -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
    preference_energy_threshold = 0  # TODO: confirm

    result_formation_energy = {"ref": []} | {mlip: [] for mlip in MODELS}
    result_spearmans_coefficient = {"ref": []} | {mlip: [] for mlip in MODELS}
    # TODO: investigate Kendall rank correlation
    result_rmsd = {mlip: [] for mlip in MODELS}

    ref_stored = False
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        for material_dir in tqdm(list(model_dir.iterdir())):
            cation_dirs = [
                p for p in material_dir.iterdir() if p.is_dir()
            ]  # skip pristine supercell.xyz files if present (not used)

            for cation_dir in tqdm(cation_dirs, leave=False):
                cation = cation_dir.stem

                nv_xyz_path = cation_dir / "normal_vacancy.xyz.gz"
                sv_xyz_path = cation_dir / "split_vacancy.xyz.gz"
                # sv_from_nv_xyz_path = cation_dir / "normal_vacancy_to_split_vac.xyz.gz"

                nv_atoms_list = read(nv_xyz_path, ":")
                sv_atoms_list = read(sv_xyz_path, ":")

                nv_energies = [at.info["relaxed_energy"] for at in nv_atoms_list]
                sv_energies = [at.info["relaxed_energy"] for at in sv_atoms_list]

                # TODO (later): result_rmsd[model_name]
                # load original structure
                # use get_rmsd

                formula = nv_atoms_list[0].info["name"]
                cell_charge = nv_atoms_list[0][-1].info["cell_charge"]  # TODO

                sv_formation_energy = min(sv_energies) - min(nv_energies)
                sv_preferred = sv_formation_energy < preference_energy_threshold

                if not ref_stored:
                    ref_nv_energies = [at.info["ref_energy"] for at in nv_atoms_list]
                    ref_sv_energies = [at.info["ref_energy"] for at in sv_atoms_list]

                    ref_sv_formation_energy = min(ref_sv_energies) - min(
                        ref_nv_energies
                    )
                    ref_sv_preferred = (
                        ref_sv_formation_energy < preference_energy_threshold
                    )

                    result_formation_energy["ref"] = ref_sv_formation_energy

                # calculate metrics
                spearmans_coefficient = spearmanr(
                    nv_energies + sv_energies, ref_sv_energies + ref_nv_energies
                )
                result_rmsd = get_rmsd()  # TODO: RMSD of what??

        ref_stored = False

    return result_formation_energy, result_spearmans_coefficient, result_rmsd
