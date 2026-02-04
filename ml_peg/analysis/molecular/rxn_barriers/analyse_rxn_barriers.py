"""Analyse CRBH20 Reaction Barriers benchmark."""

from __future__ import annotations

from pathlib import Path
import re

from ase import units
from ase.io import read, write
import pytest

# ml_peg imports
from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# --- Configuration ---
MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

# Path to where the calc script outputted the data
# Update this to match your actual folder structure
CALC_PATH = CALCS_ROOT / "molecular" / "rxn_barriers" / "outputs"

# Path where this analysis script will save data for the Streamlit App
OUT_PATH = APP_ROOT / "data" / "reaction_barriers" / "CRBH20"

# Load metrics configuration (thresholds for green/red coloring in tables)
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
# If the file doesn't exist, we provide defaults, but usually it should exist.
try:
    DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(METRICS_CONFIG_PATH)
except FileNotFoundError:
    # Fallback defaults if metrics.yml is missing
    DEFAULT_THRESHOLDS = {"MAE": [1.0, 5.0]} # Green < 1.0, Red > 5.0
    DEFAULT_TOOLTIPS = {"MAE": "Mean Absolute Error"}
    DEFAULT_WEIGHTS = {}

# --- Reference Data (Appendix B.5 of arXiv:2401.00096) ---
# Units: kcal/mol
REF_BARRIERS_KCAL = {
    1: 10.53, 2: 20.44, 3: 13.14, 4: 19.88, 5: 11.29,
    6: 11.63, 7: 13.39, 8: 10.73, 9: 11.29, 10: 12.57,
    11: 16.88, 12: 7.89,  13: 10.09, 14: 10.32, 15: 12.37,
    16: 8.16,  17: 8.88,  18: 21.74, 19: 33.92, 20: 22.62
}

def get_reaction_ids() -> list[str]:
    """
    Get list of Reaction IDs for plotting hover data.
    We just use 1..20, sorted.
    """
    return [str(i) for i in range(1, 21)]

def numeric_sort_key(filepath: Path):
    """Sort helper to ensure files 1, 2, ... 10 come in numerical order, not alpha."""
    # Extract the number from 'crbh20_12.xyz'
    match = re.search(r'crbh20_(\d+).xyz', filepath.name)
    if match:
        return int(match.group(1))
    return 0

@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_reaction_barriers.json",
    title="CRBH20 Reaction Barriers",
    x_label="Predicted Barrier (kcal/mol)",
    y_label="Reference Barrier (kcal/mol)",
    hoverdata={
        "Reaction ID": get_reaction_ids(),
    },
)
def reaction_barriers() -> dict[str, list]:
    """
    Get barriers for all CRBH20 systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted barriers in kcal/mol.
        Format: {'ref': [10.53, ...], 'mace-mp-0b3': [10.2, ...]}
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False
    
    # We iterate 1..20 to ensure the lists are perfectly aligned
    rxn_ids = range(1, 21)

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        # Temporary list to ensure we collect this model's data in 1..20 order
        model_barriers = []

        for rxn_id in rxn_ids:
            # Construct expected filename
            xyz_file = model_dir / f"crbh20_{rxn_id}.xyz"

            if not xyz_file.exists():
                # Handle missing data (e.g., if calc failed)
                # For parity plots, lists must be equal length. 
                # We append None or NaN, though ml-peg might prefer dropping the point.
                # Here we assume completeness or append 0.0 with a warning.
                model_barriers.append(None) 
                if not ref_stored: results["ref"].append(REF_BARRIERS_KCAL[rxn_id])
                continue

            # Read the combined XYZ (Reactant is index 0, TS is index 1)
            # We only need index 0 because we stored the barrier in info tag of both
            structs = read(xyz_file, index=":")
            reactant = structs[0]
            
            # Extract ML Barrier (calculated in the previous script)
            # stored as "barrier_kcal"
            barrier_ml = reactant.info.get("barrier_kcal", 0.0)
            model_barriers.append(barrier_ml)

            # Copy structure files to APP directory for visualization
            # This allows the web app to show the molecule when you hover/click
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"crbh20_{rxn_id}.xyz", structs)

            # Store reference energies (only once, during the first model loop)
            if not ref_stored:
                ref_val = REF_BARRIERS_KCAL.get(rxn_id, 0.0)
                results["ref"].append(ref_val)

        # Update the main results dict
        results[model_name] = model_barriers
        
        # Mark reference as stored so we don't duplicate it
        if any(x is not None for x in model_barriers):
            ref_stored = True

    return results

@pytest.fixture
def crbh20_errors(reaction_barriers) -> dict[str, float]:
    """
    Compute Mean Absolute Error (MAE) for reaction barriers.
    """
    results = {}
    for model_name in MODELS:
        if reaction_barriers.get(model_name):
            # Filter out None values in case of failed calculations
            y_true = []
            y_pred = []
            for r, p in zip(reaction_barriers["ref"], reaction_barriers[model_name]):
                if r is not None and p is not None:
                    y_true.append(r)
                    y_pred.append(p)
            
            if y_true:
                results[model_name] = mae(y_true, y_pred)
            else:
                results[model_name] = None
        else:
            results[model_name] = None
    return results

@pytest.fixture
@build_table(
    filename=OUT_PATH / "crbh20_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(crbh20_errors: dict[str, float]) -> dict[str, dict]:
    """
    Compile all metrics for the table.
    """
    return {
        "MAE": crbh20_errors,
    }

def test_crbh20_analysis(metrics: dict[str, dict]) -> None:
    """
    Trigger the analysis pipeline.
    
    The decorators on the fixtures above (@plot_parity, @build_table) 
    do the heavy lifting of saving the JSON files when this test runs.
    """
    # Verify we actually calculated something
    assert metrics is not None
    assert "MAE" in metrics