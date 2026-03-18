"""Run calculations for scaling_pol tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import numpy as np
import pytest
import json

from ml_peg.calcs.utils.utils import download_s3_data		# PRODUCTION
from ml_peg.calcs.utils.utils import download_github_data	# TEST
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_scaling_pol(mlip: tuple[str, Any]) -> None:
	"""
	Run scaling polarizability of linear organic molecules test.

	Parameters
	----------
	mlip
		Name of model use and model to get calculator.
	"""
	model_name, model = mlip
	clean_calc = model.get_calculator()

	# Download 'linear organic mols' dataset
	"""
	## THIS SHOULD BE USED IN PRODUCTION
	scaling_pol_dir = download_s3_data(
		filename="ORCA_DATA.zip",
		key="inputs/electric_field/scaling_pol/ORCA_DATA.zip",
	)
	"""
 
	## ONLY FOR TESTING
	scaling_pol_dir = download_github_data(
		filename="ORCA_DATA.zip",
		github_uri="https://github.com/viktorsvahn/teoroo_ML-PEG/raw/refs/heads/main/data/source",
	)

	mols = read(scaling_pol_dir/"ORCA_DATA.xyz",':')
	mol_out = []

	for mol in mols:
		calc = copy(clean_calc)
		mol.calc = calc

		energy = mol.get_potential_energy()	
		if 'dipole' in calc.results:
			mol_out.append(mol)

	# Write output structures
	if len(mol_out) > 0:
		write_dir = OUT_PATH/model_name
		write_dir.mkdir(parents=True, exist_ok=True)
		write(write_dir/"ORCA_DATA.xyz", mol_out)	# This filename needs to be adjusted to whatever's in the zipfile above

