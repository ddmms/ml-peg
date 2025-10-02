"""Run elemental slab oxygen adsorption app."""

from __future__ import annotations

import json

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.build_components import build_weight_components
from ml_peg.app.utils.load import read_plot
from ml_peg.calcs.models.models import MODELS

BENCHMARK_NAME = "Elemental Slab Oxygen Adsorption"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#elemental-slab-oxygen-adsorption"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "elemental_slab_oxygen_adsorption"


class ElementalSlabOxygenAdsorptionApp(BaseApp):
    """Elemental slab oxygen adsorption benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_adsorption_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = DATA_PATH / list(MODELS.keys())[0]

        # Assets dir will be parent directory
        structs = [
            f"assets/surfaces/elemental_slab_oxygen_adsorption/{list(MODELS.keys())[0]}/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="traj",
        )


def get_app() -> ElementalSlabOxygenAdsorptionApp:
    """
    Get elemental slab oxygen adsorption benchmark app layout and callback registration.

    Returns
    -------
    ElementalSlabOxygenAdsorptionApp
        Benchmark layout and callback registration.
    """
    # Build metric weight components (sliders + inputs) for metrics
    with open(DATA_PATH / "elemental_slab_oxygen_adsorption_metrics_table.json") as f:
        table_json = json.load(f)
    metric_columns = [
        c["id"]
        for c in table_json["columns"]
        if c["id"] not in ("MLIP", "Score", "Rank", "id")
    ]

    metric_weights = build_weight_components(
        header="Metric weights",
        columns=metric_columns,
        input_ids=[f"{BENCHMARK_NAME}-{c.replace(' ', '-')}" for c in metric_columns],
        table_id=f"{BENCHMARK_NAME}-table",
    )

    return ElementalSlabOxygenAdsorptionApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting adsorption energies of oxygen "
            "on elemental slabs."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "elemental_slab_oxygen_adsorption_metrics_table.json",
        extra_components=[
            metric_weights,
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    elemental_slab_oxygen_adsorption_app = get_app()
    full_app.layout = elemental_slab_oxygen_adsorption_app.layout
    elemental_slab_oxygen_adsorption_app.register_callbacks()

    # Run app
    full_app.run(port=8052, debug=True)
