"""Run battery electrolyte densities app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Battery Electrolyte Densities"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html"
    "#battery-electrolyte-densities"
)
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "battery_electrolyte_densities"
INFO_PATH = DATA_PATH / "info.json"


class BatteryElectrolyteDensitiesApp(BaseApp):
    """Battery electrolyte densities benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_battery_electrolyte_densities.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Keep the hover fields intact and append a model-specific asset key.
        # The shared callback resolves the last customdata field, so selection
        # does not depend on trace indices or the current model registry order.
        structs = {}
        for trace in (scatter.figure or {}).get("data", []):
            points = trace.get("customdata")
            if points is None:
                continue
            keyed_points = []
            for point in points:
                label = point[0]  # First hover field is the analysis system label.
                key = f"{trace['name']}/{label}.xyz"
                keyed_points.append([*point, key])
                if (DATA_PATH / key).is_file():
                    structs[key] = (
                        "/assets/molecular_dynamics/battery_electrolyte_densities/"
                        f"{key}"
                    )
            trace["customdata"] = keyed_points

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                **dict.fromkeys(("MAE", "RMSE", "MAPE"), scatter),
                "Stable runs": read_plot(
                    DATA_PATH / "figure_completion.json",
                    id=f"{BENCHMARK_NAME}-completion-figure",
                ),
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
            follow_frames=False,
        )


def get_app() -> BatteryElectrolyteDensitiesApp:
    """
    Get battery electrolyte densities benchmark app layout and callbacks.

    Returns
    -------
    BatteryElectrolyteDensitiesApp
        Benchmark layout and callback registration.
    """
    return BatteryElectrolyteDensitiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting densities of Na-ion battery electrolytes "
            "in glyme and carbonate solvents, and of the corresponding neat "
            "solvents. Reference data are experimental densities at 298.2 K."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "battery_electrolyte_densities_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8064, debug=True)
