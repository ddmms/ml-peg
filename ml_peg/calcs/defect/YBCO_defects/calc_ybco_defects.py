"""
Run calculations for the YBCO point-defect formation-energy benchmark.

Relaxes the perfect 6x6x2 supercell and the O2/Ba/Cu/Y references (cell + positions),
then each defect at the fixed relaxed host cell (positions only). Inputs are a zipped
bundle of LAMMPS-data files (types 1=Ba 2=Cu 3=O 4=Y).
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import read
from janus_core.calculations.geom_opt import GeomOpt
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

Z_OF_TYPE = {1: 56, 2: 29, 3: 8, 4: 39}  # Ba, Cu, O, Y

VACANCIES = ["O1", "O2", "O3", "O4", "Cu1", "Cu2", "Ba", "Y"]
ANTISITES = ["Ba_Cu1", "Ba_Cu2", "Ba_Y", "Y_Ba", "Y_Cu1", "Y_Cu2", "Cu_Y", "Cu_Ba"]
INTERSTITIALS = ["Oint1", "Oint2", "Oint3", "Oint4", "Oint5", "Oint6", "Oint7"]
REFERENCES = ["O2", "Ba", "Cu", "Y"]


def _read_struct(path: Path) -> Atoms:
    """
    Read a LAMMPS-data structure and mark it periodic and omol-safe.

    Parameters
    ----------
    path
        Path to the LAMMPS-data file.

    Returns
    -------
    Atoms
        The structure, with pbc set and charge/spin defaults in info.
    """
    atoms = read(path, format="lammps-data", Z_of_type=Z_OF_TYPE, atom_style="atomic")
    atoms.pbc = True
    atoms.info.setdefault("charge", 0)  # omol models need these; PBE ignores them
    atoms.info.setdefault("spin", 1)
    return atoms


def _relax(atoms: Atoms, calc, prefix: Path, relax_cell: bool) -> None:
    """
    Relax a structure and write its trajectory.

    Parameters
    ----------
    atoms
        Structure to relax (calculator attached by this function).
    calc
        Calculator to attach.
    prefix
        File prefix for the written trajectory.
    relax_cell
        Whether to relax the cell (True for host/references, False for defects).
    """
    atoms.calc = copy(calc)
    GeomOpt(
        struct=atoms,
        fmax=0.02,
        write_traj=True,
        file_prefix=prefix,
        filter_class=FrechetCellFilter if relax_cell else None,
    ).run()


@pytest.mark.parametrize("mlip", MODELS.items())
def test_ybco_defects(mlip: tuple[str, Any]) -> None:
    """
    Run the YBCO defect formation-energy calculations for one model.

    Parameters
    ----------
    mlip
        Name of model and model used to get a calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    out_dir = OUT_PATH / model_name

    data_dir = (
        download_s3_data(
            key="inputs/defect/YBCO_defects/YBCO_defects.zip",
            filename="YBCO_defects.zip",
        )
        / "YBCO_defects"
    )

    # perfect cell + references go in _energetics/
    energetics_dir = out_dir / "_energetics"

    # perfect: relax cell + positions, keep the relaxed cell for the defects
    perfect = _read_struct(data_dir / "perfect.data")
    try:
        _relax(perfect, calc, energetics_dir / "perfect", relax_cell=True)
    except Exception as exc:  # noqa: BLE001
        warn(f"Perfect cell relaxation failed for {model_name}: {exc}", stacklevel=2)
        return
    host_cell = perfect.cell.array.copy()

    # elemental references (for the chemical potentials)
    for ref in REFERENCES:
        ref_file = data_dir / f"ref_{ref}.data"
        if not ref_file.is_file():
            warn(f"Missing reference {ref_file}", stacklevel=2)
            continue
        try:
            _relax(
                _read_struct(ref_file),
                calc,
                energetics_dir / f"ref_{ref}",
                relax_cell=True,
            )
        except Exception as exc:
            warn(f"Reference {ref} failed for {model_name}: {exc}", stacklevel=2)

    # defects: positions only, at the fixed relaxed cell
    defect_files = (
        [("vac", d) for d in VACANCIES]
        + [("anti", d) for d in ANTISITES]
        + [("int", d) for d in INTERSTITIALS]
    )
    for kind, name in defect_files:
        struct_file = data_dir / f"{kind}_{name}.data"
        if not struct_file.is_file():
            warn(f"Missing defect {struct_file}", stacklevel=2)
            continue
        atoms = _read_struct(struct_file)
        atoms.info["name"] = name
        atoms.info["defect_class"] = {
            "vac": "vacancy",
            "anti": "antisite",
            "int": "interstitial",
        }[kind]
        atoms.set_cell(host_cell, scale_atoms=True)
        try:
            _relax(atoms, calc, out_dir / f"{kind}_{name}", relax_cell=False)
        except Exception as exc:
            warn(f"Defect {name} failed for {model_name}: {exc}", stacklevel=2)
