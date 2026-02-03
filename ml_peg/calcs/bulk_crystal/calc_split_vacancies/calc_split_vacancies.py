"""Run calculations for elasticity benchmark."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

# from pymatgen.analysis import StructureMatcher
from ase.io import read, write
from ase.optimize import LBFGS
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
# TODO: DATA_PATH = download_github_data(filename, github_uri)
DATA_PATH = Path("/u/twarford/dev/defect_data/out")
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

    for material_dir in DATA_PATH.iterdir():
        for cation_dir in material_dir.iterdir():
            if not cation_dir.is_dir():
                continue  # skip pristine supercell.xyz files (not used)

            nv_xyz_path = cation_dir / "normal_vacancy.xyz"
            sv_xyz_path = cation_dir / "split_vacancy.xyz"
            sv_from_nv_xyz_path = cation_dir / "normal_vacancy_to_split_vac.xyz"

            for atoms_path in [nv_xyz_path, sv_xyz_path, sv_from_nv_xyz_path]:
                relaxed_atoms = []
                atoms_list = read(atoms_path, ":")

                for atoms in atoms_list:
                    atoms.calc = deepcopy(calc)
                    atoms.info["initial_energy"] = atoms.get_potential_energy()

                    opt = LBFGS(atoms, logfile=None)
                    opt.run(fmax=fmax, steps=steps)

                    atoms.info["relaxed_energy"] = atoms.get_potential_energy()

                    relaxed_atoms.append(atoms)
                    break

                atoms_out_path = (
                    OUT_PATH
                    / model_name
                    / material_dir.stem
                    / cation_dir.stem
                    / f"{atoms_path.stem}.xyz.gz"
                )
                atoms_out_path.parent.mkdir(exist_ok=True, parents=True)

                write(relaxed_atoms, atoms_out_path)
                break
