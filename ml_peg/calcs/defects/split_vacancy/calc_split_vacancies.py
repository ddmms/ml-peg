"""Run calculations for split vacancy benchmark."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

# from pymatgen.analysis import StructureMatcher
from ase.io import read, write
from ase.optimize import LBFGS
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
import pytest
from tqdm.auto import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
DATA_PATH = (
    download_s3_data(
        key="inputs/defects/split_vacancy/split_vacancy.zip",
        filename="split_vacancy_data.zip",
    )
    / "split_vacancy"
)
OUT_PATH = Path(__file__).parent / "outputs"

# based on MatBench settings, see https://github.com/janosh/matbench-discovery/issues/230
# note we choose theshold stol for match in analysis
STRUCTURE_MATCHER = StructureMatcher(stol=1.0, scale=False)


def get_rms_dist(atoms_1, atoms_2) -> tuple[float, float] | None:
    """
    Calculate the RMSD between two ASE Atoms objects.

    Parameters
    ----------
    atoms_1, atoms_2
        ASE Atoms objects.

    Returns
    -------
    tuple[float, float] | None
        (Root mean square displacement, max di) or None if no match.
    """
    result = STRUCTURE_MATCHER.get_rms_dist(
        Structure.from_ase_atoms(atoms_1), Structure.from_ase_atoms(atoms_2)
    )

    if result is None:
        return (np.nan, np.nan)
    return result


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_relax_and_calculate_energy(mlip: tuple[str, Any]):
    """
    Run calculations required for split vacancy formation energies.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    fmax = 0.03
    steps = 200

    nan_counter = 0
    unconverged_counter = 0
    error_counter = 0
    for functional in ["pbe", "pbesol"]:
        for material_dir in tqdm(list((DATA_PATH / functional).iterdir())):
            cation_dirs = [
                p for p in material_dir.iterdir() if p.is_dir()
            ]  # skip pristine supercell.xyz files (not used)
            for cation_dir in tqdm(cation_dirs, leave=False):
                nv_xyz_path = cation_dir / "normal_vacancy.xyz"
                sv_xyz_path = cation_dir / "split_vacancy.xyz"

                if not (nv_xyz_path.exists() and sv_xyz_path.exists()):
                    continue

                atoms_paths = [nv_xyz_path, sv_xyz_path]

                try:
                    for atoms_path in atoms_paths:
                        relaxed_atoms = []
                        atoms_list = read(atoms_path, ":")

                        ref_atoms_out_path = (
                            OUT_PATH
                            / functional
                            / "ref"
                            / material_dir.stem
                            / cation_dir.stem
                            / f"{atoms_path.stem}.xyz"
                        )
                        # Copy ref structures once; later runs skip if present.
                        if not ref_atoms_out_path.exists():
                            ref_atoms_out_path.parent.mkdir(exist_ok=True, parents=True)
                            write(ref_atoms_out_path, atoms_list)

                        for initial_atoms in atoms_list:
                            atoms = deepcopy(initial_atoms)
                            atoms.info["charge"] = initial_atoms.info[
                                "ref_total_charge"
                            ]
                            atoms.info["spin"] = 1

                            atoms.calc = deepcopy(calc)
                            atoms.info["initial_energy"] = atoms.get_potential_energy()

                            opt = LBFGS(atoms, logfile=None)
                            opt.run(fmax=fmax, steps=steps)
                            converged = opt.converged()

                            atoms.info["relaxed_energy"] = atoms.get_potential_energy()

                            rmsd, max_dist = get_rms_dist(atoms, initial_atoms)
                            atoms.info["ref_rmsd"] = rmsd
                            atoms.info["ref_max_distance"] = max_dist
                            atoms.info["relaxation_converged"] = converged

                            relaxed_atoms.append(atoms)

                            if rmsd is np.nan:
                                nan_counter += 1
                            if not converged:
                                unconverged_counter += 1

                        atoms_out_path = (
                            OUT_PATH
                            / functional
                            / model_name
                            / material_dir.stem
                            / cation_dir.stem
                            / f"{atoms_path.stem}.xyz"
                        )
                        atoms_out_path.parent.mkdir(exist_ok=True, parents=True)
                        write(atoms_out_path, relaxed_atoms)
                except Exception as exc:
                    error_counter += 1
                    print(
                        f"Warning: skipping {material_dir.stem}/{cation_dir.stem} "
                        f"({functional}) for model {model_name}: {exc}"
                    )
                    continue
    if nan_counter > 0:
        print(
            f"Warning: {nan_counter} structures had no match with reference "
            "and were assigned NaN for RMSD and max distance for model"
            f"{model_name}. Consider increasing StructureMatcher stol in "
            "calc_split_vacancies.py."
        )
    if unconverged_counter > 0:
        print(
            f"Warning: {unconverged_counter} structures did not converge within "
            f"{steps} steps for model {model_name}. Consider increasing the "
            "number of steps or fmax in calc_split_vacancies.py."
        )
    if error_counter > 0:
        print(
            f"Warning: {error_counter} material-cation pairs were skipped due to "
            f"errors during calculation for model {model_name} (see warnings above)."
        )
