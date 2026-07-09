"""Run CRBH20 Reaction Barriers app."""

from __future__ import annotations

from pathlib import Path
import re

from dash import Dash, html

# ml-peg imports
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# --- Configuration ---
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "CRBH20 Reaction Barriers"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "molecular_reactions.html#crbh20"
)

DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "CRBH20"
INFO_PATH = DATA_PATH / "info.json"


def numeric_sort_key(filepath: Path) -> int:
    """
    Sort helper to ensure files 1, 2... 10 come in numerical order.

    Parameters
    ----------
    filepath
        Path to a structure file named crbh20_[n].xyz.

    Returns
    -------
    int
        Reaction number extracted from the filename.
    """
    match = re.search(r"crbh20_(\d+).xyz", filepath.name)
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

        # We must sort these numerically (1, 2, ... 10)
        # so they align with the plot points
        files = sorted(structs_dir.glob("*.xyz"), key=numeric_sort_key)

        # Dash Asset Paths: The 'assets' folder is mounted at APP_ROOT/data
        # So we construct the URL path starting from there.
        structs = [
            f"/assets/molecular_reactions/CRBH20/{valid_model}/{f.name}" for f in files
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
    """
    Get CRBH20 benchmark app layout and callback registration.

    Returns
    -------
    CRBH20App
        Benchmark layout and callback registration.
    """
    return CRBH20App(
        name=BENCHMARK_NAME,
        description="Reaction Barrier Heights for 20 organic reactions (kcal/mol).",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "crbh20_metrics_table.json",
        extra_components=[
            html.Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            html.Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(
        __name__,
        assets_folder=str(APP_ROOT / "data"),
        suppress_callback_exceptions=True,
    )

    # Load Layout
    app_instance = get_app()
    full_app.layout = app_instance.layout
    app_instance.register_callbacks()

    # Run Server
    print("Serving CRBH20 App on port 8055...")
    full_app.run(port=8055, debug=True)
