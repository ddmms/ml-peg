from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest

# ml_peg imports for model management
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

# Load all models defined in 'models.yml'
MODELS = load_models(current_models)

# Define standard paths for ml-peg structure
# This assumes you place your '1', '2'... folders inside ml_peg/calcs/data/CRBH20
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion: 1 eV = 23.0605 kcal/mol
EV_TO_KCAL = 23.0605

@pytest.mark.parametrize("mlip", MODELS.items())
def test_crbh20_barrier_calculation(mlip: tuple[str, Any]) -> None:
    """
    Run calculations of the reaction energy barriers for the 20 systems in CRBH20.
    
    This function will be run automatically for every model in models.yml.

    Parameters
    ----------
    mlip
        Tuple containing (model_name, model_object)
    """
    model_name, model = mlip
    
    # 1. Initialize the Calculator
    # The ml-peg wrapper handles device selection (cuda/cpu) and loading
    calc = model.get_calculator()

    # Apply D3 dispersion if the model supports/requires it (standard ml-peg pattern)
    if hasattr(model, "add_d3_calculator"):
        calc = model.add_d3_calculator(calc)

    # Create output directory for this specific model
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Rxn ID':<8} | {'Barrier (eV)':<12} | {'Barrier (kcal/mol)':<18}")
    print("="*50)

    # 2. Loop through Reaction Folders (1 to 20)
    for i in range(1, 21):
        rxn_id = str(i)
        rxn_path = DATA_PATH / rxn_id

        # Skip if data is missing (prevents crash on partial datasets)
        if not rxn_path.exists():
            continue

        energies = {}
        atoms_dict = {}

        # 3. Calculate Energy for Reactant and Transition State
        for state in ['react', 'ts']:
            poscar_path = rxn_path / state / "POSCAR"
            
            if poscar_path.exists():
                # Read geometry
                atoms = read(poscar_path)
                atoms.calc = calc
                
                # Compute Energy
                e_pot = atoms.get_potential_energy()
                
                energies[state] = e_pot
                atoms_dict[state] = atoms
                
                # Store metadata in atoms.info (useful for the output xyz file)
                atoms.info["rxn_id"] = rxn_id
                atoms.info["state"] = state
                atoms.info["energy_ev"] = e_pot
                atoms.info["model"] = model_name

        # 4. Compute Barrier and Write Output
        if 'react' in energies and 'ts' in energies:
            barrier_ev = energies['ts'] - energies['react']
            barrier_kcal = barrier_ev * EV_TO_KCAL
            
            # Log to console
            print(f"{rxn_id:<8} | {barrier_ev:.4f}       | {barrier_kcal:.4f}")
            
            # Tag both structures with the calculated barrier
            for state in ['react', 'ts']:
                atoms_dict[state].info["barrier_ev"] = barrier_ev
                atoms_dict[state].info["barrier_kcal"] = barrier_kcal

            # Write combined XYZ file (Reactant + TS) for this reaction
            # This creates: ml_peg/calcs/outputs/mace-mp-0b3/crbh20_1.xyz
            output_filename = write_dir / f"crbh20_{rxn_id}.xyz"
            write(output_filename, [atoms_dict['react'], atoms_dict['ts']])

# """Run calculations of reaction energy barriers for 20 systems"""

# from __future__ import annotations

# from copy import copy
# from pathlib import Path
# from typing import Any

# from ase import units
# from ase.io import read, write
# import numpy as np
# import pytest

# from ml_peg.calcs.utils.utils import download_s3_data
# from ml_peg.models.get_models import load_models
# from ml_peg.models.models import current_models

# MODELS = load_models(current_models)

# DATA_PATH = Path(__file__).parent / "data"
# OUT_PATH = Path(__file__).parent / "outputs"

# # Unit conversion
# EV_TO_KJ_PER_MOL = units.mol / units.kJ

# @pytest.mark.parametrize("mlip", MODELS.items())
# def test_rxn_barrier_calculation(mlip: tuple[str, Any]) -> None:
#     """
#     Run calculations of the reaction energy barriers for the 20 systems in CRBH20.

#     Parameters
#     ----------
#     mlip
#         Name of model use and model to get calculator.
#     """
#     model_name, model = mlip
#     calc = model.get_calculator()

#     # Add D3 calculator for this test (for models where applicable)
#     calc = model.add_d3_calculator(calc)

#     # Download X23 dataset
#     lattice_energy_dir = (
#         download_s3_data(
#             key="inputs/molecular_crystal/X23/X23.zip",
#             filename="lattice_energy.zip",
#         )
#         / "lattice_energy"
#     )

#     with open(lattice_energy_dir / "list") as f:
#         systems = f.read().splitlines()

#     for system in systems:
#         molecule_path = lattice_energy_dir / system / "POSCAR_molecule"
#         solid_path = lattice_energy_dir / system / "POSCAR_solid"
#         ref_path = lattice_energy_dir / system / "lattice_energy_DMC"
#         num_molecules_path = lattice_energy_dir / system / "nmol"

#         molecule = read(molecule_path, index=0, format="vasp")
#         molecule.calc = calc
#         molecule.get_potential_energy()

#         solid = read(solid_path, index=0, format="vasp")
#         solid.calc = copy(calc)
#         solid.get_potential_energy()

#         ref = np.loadtxt(ref_path)[0]
#         num_molecules = np.loadtxt(num_molecules_path)

#         solid.info["ref"] = ref
#         solid.info["num_molecules"] = num_molecules
#         solid.info["system"] = system
#         molecule.info["ref"] = ref
#         molecule.info["num_molecules"] = num_molecules
#         molecule.info["system"] = system

#         # Write output structures
#         write_dir = OUT_PATH / model_name
#         write_dir.mkdir(parents=True, exist_ok=True)
#         write(write_dir / f"{system}.xyz", [solid, molecule])




# import os
# from mace.calculators import mace_mp
# from ase.io import read

# # 1. Initialize the model ONCE (much faster than reloading it 40 times)
# # Define the path to your downloaded model
# # Make sure the filename matches exactly what you downloaded
# model_path = os.path.abspath("models/mace-mp-0b3-medium.model")

# print(f"Loading MACE model from: {model_path}")

# # Load the model using the path
# macemp = mace_mp(
#     model=model_path,     # <--- Point to the local file here
#     dispersion=True,      # Keep dispersion on
#     default_dtype="float64",
#     device="cuda"         # Use "cpu" if you don't have a GPU
# )

# # Conversion: 1 eV = 23.0605 kcal/mol
# EV_TO_KCAL = 23.0605

# print(f"{'Rxn ID':<8} | {'Barrier (eV)':<12} | {'Barrier (kcal/mol)':<18}")
# print("="*50)

# # 2. Loop through all reaction folders (assuming 1 to 20)
# for i in range(1, 21):
#     rxn_id = str(i)
#     energies = {}
    
#     # Check both Reactant and Transition State
#     for state in ['react', 'ts']:
#         # Construct the path: e.g., "1/react/POSCAR"
#         path = os.path.join(rxn_id, state, 'POSCAR')
        
#         if os.path.exists(path):
#             try:
#                 # Read the geometry
#                 atoms = read(path)
                
#                 # Attach the calculator we loaded earlier
#                 atoms.calc = macemp
                
#                 # Calculate Potential Energy
#                 e_pot = atoms.get_potential_energy()
#                 energies[state] = e_pot
                
#                 # OPTIONAL: Write to file to save progress
#                 # This overwrites ('w') to prevent duplicate lines if you re-run
#                 output_file = os.path.join(rxn_id, state, 'energy-mace')
#                 with open(output_file, "w") as f:
#                     print(e_pot, file=f)
                    
#             except Exception as e:
#                 print(f"Error calculating {path}: {e}")
#         else:
#             # Silently skip if folder doesn't exist (useful if you only have some folders)
#             pass

#     # 3. Calculate and Print Barrier
#     if 'react' in energies and 'ts' in energies:
#         barrier_ev = energies['ts'] - energies['react']
#         barrier_kcal = barrier_ev * EV_TO_KCAL
#         print(f"{rxn_id:<8} | {barrier_ev:.4f}       | {barrier_kcal:.4f}")
#     else:
#         print(f"{rxn_id:<8} | {'MISSING DATA':<30}")