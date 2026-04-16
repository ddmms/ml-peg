"""Run Defectstab app."""

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
BENCHMARK_NAME = "Defectstab Formation Energies"
# Update this URL when documentation is added
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/defect.html#defectstab"
DATA_PATH = APP_ROOT / "data" / "defect" / "Defectstab"


class DefectstabApp(BaseApp):
    """Defectstab benchmark app layout and callbacks."""

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
                filename = "defectstab_metrics_table.json"
            else:
                filename = f"defectstab_metrics_table_{subset_value}.json"

            try:
                new_table = rebuild_table(
                    DATA_PATH / filename,
                    id="temp-id-not-used",
                    description=self.description,
                )

                data = new_table.data

                if current_weights:
                    data = calc_table_scores(
                        data,
                        weights=current_weights,
                        thresholds=new_table.thresholds,
                    )

                scored_data = calc_metric_scores(data, new_table.thresholds)
                style = get_table_style(data, scored_data=scored_data)

                return data, style
            except FileNotFoundError:
                return [], []

        scatter = read_plot(
            DATA_PATH / "figure_energy.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        # Assuming the first model has all structures
        structs_dir = DATA_PATH / MODELS[0]
        # Sort files to match the order in the scatter plot
        structs = [
            f"/assets/defect/Defectstab/{MODELS[0]}/{struct_file.name}"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"RMSD": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> DefectstabApp:
    """
    Get Defectstab benchmark app layout and callback registration.

    Returns
    -------
    DefectstabApp
        Benchmark layout and callback registration.
    """
    # Build dropdown options from per-subset table files
    subset_files = sorted(DATA_PATH.glob("defectstab_metrics_table_*.json"))
    options = [{"label": "Average (Total)", "value": "total"}]
    for p in subset_files:
        subset = p.stem.replace("defectstab_metrics_table_", "")
        options.append({"label": f"Subset: {subset}", "value": subset})

    dropdown = dcc.Dropdown(
        id=f"{BENCHMARK_NAME}-subset-dropdown",
        options=options,
        value="total",
        clearable=False,
        style={"marginBottom": "10px", "width": "50%"},
    )

    return DefectstabApp(
        name=BENCHMARK_NAME,
        description="Formation energies of point defects.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "defectstab_metrics_table.json",
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
    defectstab_app = get_app()
    full_app.layout = defectstab_app.layout
    defectstab_app.register_callbacks()

    # Run app
    full_app.run(port=8055, debug=True)
