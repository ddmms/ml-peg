"""Run calculations for split vacancy benchmark."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

# from pymatgen.analysis import StructureMatcher
from ase.io import read, write
from ase.optimize import LBFGS
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
import pytest
from tqdm.auto import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
# TODO: DATA_PATH = download_github_data(filename, github_uri)
DATA_PATH = Path("/Users/tw/Downloads/split_vacancy_data")
OUT_PATH = Path(__file__).parent / "outputs"

# same setting as MatBench
# https://github.com/janosh/matbench-discovery/blob/93cc6907ac08b4adaa8391ccc4adf9c015c0dd61/matbench_discovery/structure/symmetry.py#L124
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
    return STRUCTURE_MATCHER.get_rms_dist(
        Structure.from_ase_atoms(atoms_1), Structure.from_ase_atoms(atoms_2)
    )


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
    model.default_dtype = "float64"
    calc = model.get_calculator()

    fmax = 0.03
    steps = 200

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

                for atoms_path in tqdm(atoms_paths, leave=False):
                    relaxed_atoms = []
                    atoms_list = read(atoms_path, ":")

                    ref_atoms_out_path = (
                        OUT_PATH
                        / functional
                        / "ref"
                        / material_dir.stem
                        / cation_dir.stem
                        / f"{atoms_path.stem}.xyz.gz"
                    )
                    # Copy ref structures once; subsequent runs skip if already present.
                    if not ref_atoms_out_path.exists():
                        ref_atoms_out_path.parent.mkdir(exist_ok=True, parents=True)
                        write(ref_atoms_out_path, atoms_list)

                    for initial_atoms in tqdm(atoms_list, leave=False):
                        atoms = deepcopy(initial_atoms)
                        atoms.calc = deepcopy(calc)
                        atoms.info["initial_energy"] = atoms.get_potential_energy()

                        opt = LBFGS(atoms, logfile=None)
                        opt.run(fmax=fmax, steps=steps)

                        atoms.info["relaxed_energy"] = atoms.get_potential_energy()

                        rmsd_max_dist = get_rms_dist(atoms, initial_atoms)
                        if rmsd_max_dist is None:
                            atoms.info["ref_structure_match"] = False
                        else:
                            rmsd, max_dist = rmsd_max_dist
                            atoms.info["ref_structure_match"] = True
                            atoms.info["ref_rmsd"] = rmsd
                            atoms.info["ref_max_distance"] = max_dist

                        relaxed_atoms.append(atoms)

                    atoms_out_path = (
                        OUT_PATH
                        / functional
                        / model_name
                        / material_dir.stem
                        / cation_dir.stem
                        / f"{atoms_path.stem}.xyz.gz"
                    )
                    atoms_out_path.parent.mkdir(exist_ok=True, parents=True)

                    write(atoms_out_path, relaxed_atoms)
