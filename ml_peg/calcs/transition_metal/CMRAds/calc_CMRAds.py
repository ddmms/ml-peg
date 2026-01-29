"""Run calculations for S24 benchmark."""

from __future__ import annotations

from copy import copy
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
import pandas as pd
import pytest
import numpy as np

from ml_peg.calcs.utils.utils import chdir, download_github_data

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

@pytest.mark.parametrize("mlip", MODELS.items())
def test_CMRAds(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the CMRAds dataset.

    Adsorption energy calculations for 200 systems involving 8 different adsorbates on 25 different transition metal surfaces at full coverage. References values are taken from https://cmr.fysik.dtu.dk/adsorption/adsorption.html#adsorption. Cite: J. Phys. Chem. C 2018, 122, 8, 4381–4390.

    - Each system consists of surface, molecule+surface, adsorbate molecule and in some cases a product molecule:
        1. H2O + slab -> OH/slab + 1/2 H2
        2. CH4 + slab -> CH/slab + 3/2 H2
        3. NO + slab -> NO/slab
        4. CO + slab -> CO/slab
        5. N2 + slab -> N2/slab
        6. 1/2 N2 + slab -> N/slab
        7. 1/2 O2 + slab -> O/slab
        8. 1/2 H2 + slab -> H/slab
    - Computes adsorption energy = E(sum of products on the right) - E(reactants on the left)

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """

    model_name, model = mlip

    # Use double precision for this test
    model.default_dtype = "float64"
    calc = model.get_calculator()

    data_dir = download_github_data(filename="CMR.zip", github_uri="https://github.com/benshi97/ml-peg-transition-metal-data/raw/main", force = True)
    
    cmr_structs_list = read( data_dir / "CMRads_structures.extxyz", index = ':' )
    cmrads_reactions_dict = (pd.read_csv(data_dir / "reactions_dict.csv", index_col=0).to_dict(orient="index"))

    # Add calc to the systems in cmr_structs_list
    for atoms in cmr_structs_list:
        atoms.calc = copy(calc)

    structs_energy_dict = { struct.info['sys_formula'] : struct for struct in cmr_structs_list }

    # Compute adsorption energies
    for reaction_id in cmrads_reactions_dict.keys():
        systems_list = cmrads_reactions_dict[ reaction_id ]['Systems'].split(",")
        stoichiometry_list = np.array(cmrads_reactions_dict[ reaction_id ]['Stoichiometry'].split(",")).astype(float)
        systems_energies = np.array([ structs_energy_dict[ sys ].get_potential_energy() for sys in systems_list ])
        reaction_energy = np.sum( systems_energies * stoichiometry_list )
        print(f"Reaction ID: {reaction_id}, Predicted Reaction Energy: {reaction_energy:.4f} eV, Reference Reaction Energy: {cmrads_reactions_dict[ reaction_id ]['PBE_ads']:.4f} eV")
        mol_surface = structs_energy_dict[ systems_list[0] ].copy()
        mol_surface.info[f"pred_adsorption_energy"] = reaction_energy
        mol_surface.info["PBE_adsorption_energy"] = cmrads_reactions_dict[ reaction_id ]['PBE_ads']
        mol_surface.info["RPA_adsorption_energy"] = cmrads_reactions_dict[ reaction_id ]['RPA_ads']
        mol_surface.calc = None
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{reaction_id}.extxyz", mol_surface)
