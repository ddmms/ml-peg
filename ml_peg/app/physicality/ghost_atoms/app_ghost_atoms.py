"""Run ghost atoms app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import struct_from_table
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Ghost Atoms"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#ghost-atoms"
)
DATA_PATH = APP_ROOT / "data" / "physicality" / "ghost_atoms"


class GhostAtomsApp(BaseApp):
    """Ghost atoms benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Assets dir will be parent directory - individual files for each system
        assets_dir = f"assets/physicality/ghost_atoms/{MODELS[0]}"
        structs = {
            "Ghost atoms max ΔF": f"{assets_dir}/system_ghost.xyz",
            "Random hydrogen max ΔF": f"{assets_dir}/system_random_H.xyz",
            "Random hydrogen std ΔF": f"{assets_dir}/system_random_H.xyz",
        }

        struct_from_table(
            table_id=self.table_id,
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            column_to_struct=structs,
            mode="traj",
        )


def get_app() -> GhostAtomsApp:
    """
    Get ghost atoms benchmark app layout and callback registration.

    Returns
    -------
    GhostAtomsApp
        Benchmark layout and callback registration.
    """
    return GhostAtomsApp(
        name=BENCHMARK_NAME,
        description="Force sensitivity for ghost atoms and randomly place hydrogens.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ghost_atoms_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    ghost_atoms_app = get_app()
    full_app.layout = ghost_atoms_app.layout
    ghost_atoms_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
