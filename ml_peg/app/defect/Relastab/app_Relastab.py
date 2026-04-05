"""Run Relastab app."""

from __future__ import annotations

from dash import Dash, Input, Output, State, callback, dcc
from dash.html import Div

from ml_peg.analysis.utils.utils import (
    calc_metric_scores,
    calc_table_scores,
    get_table_style,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot, rebuild_table
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Relastab Relative Stability"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/defect.html#relastab"
DATA_PATH = APP_ROOT / "data" / "defect" / "Relastab"


class RelastabApp(BaseApp):
    """Relastab benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""

        # Callback to update table data based on subset selection
        @callback(
            Output(self.table_id, "data"),
            Output(self.table_id, "style_data_conditional"),
            Input(f"{BENCHMARK_NAME}-subset-dropdown", "value"),
            State(f"{self.table_id}-weight-store", "data"),
        )
        def update_table_data(
            subset_value: str, current_weights: dict
        ) -> tuple[list[dict], list[dict]]:
            """
            Update displayed table when subset selection changes.

            Parameters
            ----------
            subset_value
                Selected subset name or 'total'.
            current_weights
                Current metric weights from the UI.

            Returns
            -------
            tuple[list[dict], list[dict]]
                Table data and conditional styles.
            """
            if subset_value == "total":
                filename = "relastab_metrics_table.json"
            else:
                filename = f"relastab_metrics_table_{subset_value}.json"

            try:
                # Rebuild table to get the processed data (padding, scoring etc)
                new_table = rebuild_table(
                    DATA_PATH / filename,
                    id="temp-id-not-used",
                    description=self.description,
                )

                data = new_table.data

                # Re-apply current weights from UI if available.
                # Uses default scores from analysis on initial load.
                if current_weights:
                    # new_table object has the thresholds loaded from JSON
                    data = calc_table_scores(
                        data, weights=current_weights, thresholds=new_table.thresholds
                    )

                # Need to update styles because data values have changed
                # (either from new file or new weights affecting Score)
                scored_data = calc_metric_scores(data, new_table.thresholds)
                style = get_table_style(data, scored_data=scored_data)

                return data, style
            except FileNotFoundError:
                # Return empty if file doesn't exist.
                return [], []

        scatter = read_plot(
            DATA_PATH / "figure_energy.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assuming first model has structure files
        structs_dir = DATA_PATH / MODELS[0]
        # Sort glob to match analysis order
        structs = [
            f"assets/defect/Relastab/{MODELS[0]}/{struct_path.name}"
            for struct_path in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"GlobalMin": scatter, "Top5_Spearman": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> RelastabApp:
    """
    Get Relastab benchmark app.

    Returns
    -------
    RelastabApp
        Benchmark layout and callback registration.
    """
    # Build dropdown options
    subset_files = sorted(DATA_PATH.glob("relastab_metrics_table_*.json"))
    options = [{"label": "Average (Total)", "value": "total"}]
    for p in subset_files:
        subset = p.stem.replace("relastab_metrics_table_", "")
        options.append({"label": f"Subset: {subset}", "value": subset})

    dropdown = dcc.Dropdown(
        id=f"{BENCHMARK_NAME}-subset-dropdown",
        options=options,
        value="total",
        clearable=False,
        style={"marginBottom": "10px", "width": "50%"},
    )

    return RelastabApp(
        name=BENCHMARK_NAME,
        description="Relative stability ranking of defect configurations.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "relastab_metrics_table.json",
        extra_components=[
            dropdown,
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    rel_app = get_app()
    full_app.layout = rel_app.layout
    rel_app.register_callbacks()

    # Run app
    full_app.run(port=8055, debug=True)
