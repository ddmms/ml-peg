"""
Minimize reference geometries of small organic molecules.

Short energy minimizations of neutral and charged organic molecules from the
OpenFF industry dataset, measuring how far the relaxed geometry drifts from the
reference structure.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from time import perf_counter
from typing import Any
from warnings import warn

from ase import Atoms
from ase.optimize import BFGS
import numpy as np
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlip.simulation import SimulationState
from mlipaudit.benchmarks.reference_geometry_stability import (
    reference_geometry_stability as rgs,
)
from mlipaudit.io import write_model_output_to_disk

from ml_peg.calcs.utils.mlipaudit import MlPegReferenceGeometryStabilityBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


def minimize_molecule(molecule, calc) -> SimulationState:
    """
    Minimize one molecule using the audit BFGS protocol.

    Parameters
    ----------
    molecule
        Reference geometry and molecular charge.
    calc
        ASE calculator used for minimization.

    Returns
    -------
    SimulationState
        Final coordinates, energy, forces and optimizer step.
    """
    atoms = Atoms(molecule.atom_symbols, positions=molecule.coordinates)
    atoms.info.update(charge=molecule.charge, spin=rgs.DEFAULT_SPIN)
    atoms.calc = calc
    start = perf_counter()
    with BFGS(atoms, logfile=None) as optimizer:
        if not optimizer.run(
            fmax=rgs.SIMULATION_CONFIG["max_force_convergence_threshold"],
            steps=rgs.SIMULATION_CONFIG["num_steps"],
        ):
            raise ValueError("Geometry optimization did not converge")
        return SimulationState(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions()[None, ...],
            forces=atoms.get_forces()[None, ...],
            potential_energy=np.array([atoms.get_potential_energy()]),
            step=optimizer.nsteps,
            compute_time_seconds=perf_counter() - start,
        )


@pytest.mark.medium
@pytest.mark.framework("mlip_audit")
@pytest.mark.parametrize("mlip", MODELS.items())
def test_reference_geometry_stability(mlip: tuple[str, Any]) -> None:
    """
    Benchmark reference geometry stability.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/molecular/reference_geometry_stability/reference_geometry_stability.zip",
        filename="reference_geometry_stability.zip",
    )
    benchmark_name = MlPegReferenceGeometryStabilityBenchmark.name
    shutil.copytree(
        data_input_dir / benchmark_name,
        OUT_PATH / benchmark_name,
        dirs_exist_ok=True,
    )

    benchmark = MlPegReferenceGeometryStabilityBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    output = rgs.ReferenceGeometryStabilityModelOutput(
        openff_neutral=[], openff_charged=[]
    )
    for dataset, records in (
        (benchmark._openff_neutral_dataset, output.openff_neutral),
        (benchmark._openff_charged_dataset, output.openff_charged),
    ):
        for name, molecule in dataset.items():
            try:
                state = minimize_molecule(molecule, calc)
            except Exception as exc:
                warn(f"{model_name}/{name}: {exc}", stacklevel=2)
                state = None
            records.append(
                rgs.MoleculeModelOutput(
                    molecule_name=name, simulation_state=state, failed=state is None
                )
            )

    write_model_output_to_disk(
        MlPegReferenceGeometryStabilityBenchmark.name,
        output,
        OUT_PATH / model_name,
    )
