"""Analyse ghost atoms benchmark."""

from __future__ import annotations

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "ghost_atoms" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "ghost_atoms"

DEFAULT_THRESHOLDS = {
    "Ghost atoms max ΔF": (0, 5.0),
    "Random hydrogen mean ΔF": (0, 5.0),
    "Random hydrogen std ΔF": (0.0, 5.0),
}


@pytest.fixture
def ghost_force() -> dict[str, float]:
    """
    Get maximum difference in force between solute and solute with ghost atoms.

    Returns
    -------
    dict[str, float]
        Dictionary of maximum force differences for all models.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in MODELS:
        solute, combined = read(CALC_PATH / model_name / "system_ghost.xyz", index=":")

        solute_force = solute.get_forces()
        combined_force = combined.get_forces()

        # Get difference in forces on solute atoms only
        delta_f = np.linalg.norm(solute_force - combined_force[: len(solute)], axis=1)
        # Convert to meV/Å
        results[model_name] = np.max(delta_f * 1000)

        # Write structures in order as glob is unsorted
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / "system_ghost.xyz", [solute, combined])

    return results


@pytest.fixture
def hydrogen_force() -> dict[str, float]:
    """
    Get mean and std of force difference between solute and solute with hydrogen atoms.

    Returns
    -------
    dict[str, float]
        Dictionary of mean and standard deviation of force differences for all models.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in MODELS:
        structs = read(CALC_PATH / model_name / "system_random_H.xyz", index=":")

        # First structuer is pure solute
        solute_force = structs[0].get_forces()

        delta_forces = []
        for struct in structs[1:]:
            combined_force = struct.get_forces()
            # Calculate difference in forces on solute atoms only
            delta_f = np.linalg.norm(
                solute_force - combined_force[: len(structs[0])], axis=1
            )
            # Convert to meV/Å
            delta_forces.append(np.mean(delta_f) * 1000)

        results[model_name] = np.mean(delta_forces), np.std(delta_forces)

        # Write structures in order as glob is unsorted
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / "system_random_H.xyz", structs)

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ghost_atoms_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "Ghost atoms max ΔF": "Maximum force difference on solute atoms due to ghost "
        "atoms (meV/Å))",
        "Random hydrogen mean ΔF": "Mean force difference on solute atoms due to "
        "random hydrogen atoms (meV/Å)",
        "Random hydrogen std ΔF": "Standard deviation of force difference on solute "
        "atoms due to random hydrogen atoms (meV/Å)",
    },
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    ghost_force: dict[str, float], hydrogen_force: dict[str, tuple[float, float]]
) -> dict[str, dict]:
    """
    Get all ghost atom metrics.

    Parameters
    ----------
    ghost_force
        Maximum force difference between solute and solute with ghost atoms for all
        models.
    hydrogen_force
        Mean and standard deviation of force difference between solute and solute with
        hydrogen atoms for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Ghost atoms max ΔF": ghost_force,
        "Random hydrogen mean ΔF": {
            model: value[0] for model, value in hydrogen_force.items()
        },
        "Random hydrogen std ΔF": {
            model: value[1] for model, value in hydrogen_force.items()
        },
    }


def test_ghost_atoms(metrics: dict[str, dict]) -> None:
    """
    Run ghost atoms analysis.

    Parameters
    ----------
    metrics
        All ghost atoms metrics.
    """
    return
