"""
Run calculations for the YBCO lattice-parameters-vs-oxygen-content benchmark.

For each oxygen content, the 4x4x2 supercell is relaxed (cell + positions); a, b, c are
calculated in the analysis stage as the supercell lengths over the repetitions. Inputs
are LAMMPS-data files YBCO<conc>.data (types 1=Ba 2=Cu 3=O 4=Y), one per oxygen content.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read
from janus_core.calculations.geom_opt import GeomOpt
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# LAMMPS-data atom type -> atomic number (Ba, Cu, O, Y).
Z_OF_TYPE = {1: 56, 2: 29, 3: 8, 4: 39}
# Supercell repetitions the input structures were built with (a, b, c).
REPS = (4, 4, 2)
# Oxygen contents 6.00 .. 7.00.
CONCENTRATIONS = [round(6.0 + 0.1 * i, 2) for i in range(11)]


@pytest.mark.parametrize("mlip", MODELS.items())
def test_ybco_lattice_vs_O(mlip: tuple[str, Any]) -> None:  # noqa: N802
    """
    Relax YBa2Cu3O(6+x) supercells across oxygen content for one model.

    Parameters
    ----------
    mlip
        Name of model and model used to get a calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_dir = (
        download_s3_data(
            key="inputs/bulk_crystal/YBCO_lattice_vs_O/YBCO_lattice_vs_O.zip",
            filename="YBCO_lattice_vs_O.zip",
        )
        / "YBCO_lattice_vs_O"
    )

    for conc in CONCENTRATIONS:
        struct_file = data_dir / f"YBCO{conc:.2f}.data"
        if not struct_file.is_file():
            warn(f"Missing input structure {struct_file}", stacklevel=2)
            continue

        atoms = read(
            struct_file,
            format="lammps-data",
            Z_of_type=Z_OF_TYPE,
            atom_style="atomic",
        )
        atoms.pbc = True
        atoms.info["conc"] = conc
        atoms.info["reps"] = list(REPS)
        atoms.info["name"] = f"YBCO{conc:.2f}"  # scatter-point label
        atoms.info.setdefault(
            "charge", 0
        )  # same as DFT settings, omol models need these; PBE models ignore
        atoms.info.setdefault("spin", 1)

        atoms.calc = copy(calc)
        try:
            # GeomOpt relaxes the cell by default
            GeomOpt(
                struct=atoms,
                fmax=0.01,
                write_traj=True,
                file_prefix=OUT_PATH / model_name / f"YBCO{conc:.2f}",
            ).run()
        except Exception as exc:
            warn(f"Error relaxing YBCO{conc:.2f} for {model_name}: {exc}", stacklevel=2)
