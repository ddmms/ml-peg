"""Run CRBH20 Reaction Barriers app."""

from __future__ import annotations
import re
from dash import Dash, html
from pathlib import Path

# ml-peg imports
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# --- Configuration ---
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "CRBH20 Reaction Barriers"
# Placeholder docs link
DOCS_URL = "https://github.com/ddmms/ml-peg"

# Point to where we saved the JSONs (ml_peg/app/data/reaction_barriers/CRBH20)
DATA_PATH = APP_ROOT / "data" / "reaction_barriers" / "CRBH20"

def numeric_sort_key(filepath: Path):
    """Sort helper to ensure files 1, 2... 10 come in numerical order."""
    match = re.search(r'crbh20_(\d+).xyz', filepath.name)
    return int(match.group(1)) if match else 0

class CRBH20App(BaseApp):
    """CRBH20 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        
        # 1. Load the Parity Plot Data
        scatter = read_plot(
            DATA_PATH / "figure_reaction_barriers.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # 2. Setup Structure Visualization
        # We pick the first available model to source the structures from
        # (Since all models have the same geometry for the reactant/TS generally)
        # Note: We loop through MODELS to find one that actually exists in your data
        valid_model = MODELS[0]
        for m in MODELS:
            if (DATA_PATH / m).exists():
                valid_model = m
                break
        
        structs_dir = DATA_PATH / valid_model
        
        # We must sort these numerically (1, 2, ... 10) so they align with the plot points
        files = sorted(structs_dir.glob("*.xyz"), key=numeric_sort_key)
        
        # Dash Asset Paths: The 'assets' folder is mounted at APP_ROOT/data
        # So we construct the URL path starting from there.
        structs = [
            f"assets/reaction_barriers/CRBH20/{valid_model}/{f.name}"
            for f in files
        ]

        # 3. Link Table Rows to the Plot
        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        # 4. Link Plot Points to the Structure Viewer
        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )

def get_app() -> CRBH20App:
    """Get CRBH20 benchmark app layout."""
    return CRBH20App(
        name=BENCHMARK_NAME,
        description="Reaction Barrier Heights for 20 organic reactions (kcal/mol).",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "crbh20_metrics_table.json",
        extra_components=[
            html.Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            html.Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )

if __name__ == "__main__":
    # Initialize Dash
    # IMPORTANT: We set assets_folder to 'ml_peg/app/data' so it finds your JSONs/XYZs
    full_app = Dash(
        __name__, 
        assets_folder=str(APP_ROOT / "data"),
        suppress_callback_exceptions=True
    )

    # Load Layout
    app_instance = get_app()
    full_app.layout = app_instance.layout
    app_instance.register_callbacks()

    # Run Server
    print(f"Serving CRBH20 App on port 8055...")
    full_app.run(port=8055, debug=True)