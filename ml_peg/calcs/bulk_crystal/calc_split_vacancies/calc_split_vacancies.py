"""Run calculations for elasticity benchmark."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

# from pymatgen.analysis import StructureMatcher
from ase.io import read, write
from ase.optimize import LBFGS
import pytest
from tqdm.auto import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
# TODO: DATA_PATH = download_github_data(filename, github_uri)
DATA_PATH = Path("/Users/tw/Downloads/out")
OUT_PATH = Path(__file__).parent / "outputs"


# def rmsd(atoms_1, atoms_2):


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
    calc = model.get_calculator()

    fmax = 0.03
    steps = 200

    for material_dir in tqdm(list(DATA_PATH.iterdir())):
        cation_dirs = [
            p for p in material_dir.iterdir() if p.is_dir()
        ]  # skip pristine supercell.xyz files (not used)
        for cation_dir in tqdm(cation_dirs, leave=False):
            nv_xyz_path = cation_dir / "normal_vacancy.xyz"
            sv_xyz_path = cation_dir / "split_vacancy.xyz"
            sv_from_nv_xyz_path = cation_dir / "normal_vacancy_to_split_vac.xyz"

            for atoms_path in tqdm(
                [nv_xyz_path, sv_xyz_path, sv_from_nv_xyz_path], leave=False
            ):
                if not atoms_path.exists():
                    continue

                relaxed_atoms = []
                atoms_list = read(atoms_path, ":")

                for atoms in tqdm(atoms_list, leave=False):
                    atoms.calc = deepcopy(calc)
                    atoms.info["initial_energy"] = atoms.get_potential_energy()

                    opt = LBFGS(atoms, logfile=None)
                    opt.run(fmax=fmax, steps=steps)

                    atoms.info["relaxed_energy"] = atoms.get_potential_energy()

                    relaxed_atoms.append(atoms)
                    # break

                atoms_out_path = (
                    OUT_PATH
                    / model_name
                    / material_dir.stem
                    / cation_dir.stem
                    / f"{atoms_path.stem}.xyz.gz"
                )
                atoms_out_path.parent.mkdir(exist_ok=True, parents=True)

                write(atoms_out_path, relaxed_atoms)
                # break
