"""Run Ti64 phonon dispersion + DOS + TP app."""

from __future__ import annotations

from functools import partial
import json

from dash import Dash, dcc, html

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.bulk_crystal.ti64_phonons.ti64_interactive_helpers import (
    lookup_system_entry,
    render_dispersion_component,
)
from ml_peg.app.utils.build_callbacks import (
    model_asset_from_scatter,
    scatter_and_assets_from_table,
)
from ml_peg.app.utils.plot_helpers import (
    build_serialized_scatter_content,
    resolve_scatter_selection,
)

BENCHMARK_NAME = "ti64_phonons"

DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / BENCHMARK_NAME
TABLE_PATH = DATA_PATH / "ti64_phonons_metrics_table.json"
SCATTER_PATH = DATA_PATH / "ti64_phonons_interactive.json"

CALC_ROOT = APP_ROOT.parent / "calcs" / "bulk_crystal" / BENCHMARK_NAME

DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#phonons"
)

PLOT_CONTAINER_ID = f"{BENCHMARK_NAME}-plot-container"
DISPERSION_CONTAINER_ID = f"{BENCHMARK_NAME}-dispersion-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_NAME}-last-cell"
SCATTER_METADATA_STORE_ID = f"{BENCHMARK_NAME}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_NAME}-scatter"


class Ti64PhononsApp(BaseApp):
    """Ti64 phonons benchmark app wiring callbacks and layout."""

    def register_callbacks(self) -> None:
        """Register scatter/dispersion callbacks via shared helpers."""
        with SCATTER_PATH.open(encoding="utf8") as handle:
            interactive_data = json.load(handle)

        models_data = interactive_data.get("models", {})

        metric_labels = interactive_data.get("metrics", {})  # {metric_id: label}
        label_to_key = {label: key for key, label in metric_labels.items()}

        omega_label = metric_labels.get("omega_avg_thz_mae", "ω_avg MAE")

        metric_handler = partial(
            build_serialized_scatter_content,
            models_data=models_data,
            label_map=label_to_key,
            scatter_id=SCATTER_GRAPH_ID,
            instructions="Click any cell to view ω_avg (ref vs pred) scatter.",
        )

        def omega_only_handler(model_display: str, column_id: str):
            """
            Render the ω_avg scatter for the selected model.

            Parameters
            ----------
            model_display
                Display name of the selected model row.
            column_id
                Column identifier from the table callback.

            Returns
            -------
            Any
                Dash component(s) produced by the scatter renderer.
            """
            _ = column_id
            return metric_handler(model_display, omega_label)

        column_handlers = dict.fromkeys(label_to_key.keys(), omega_only_handler)

        scatter_and_assets_from_table(
            table_id=self.table_id,
            table_data=self.table.data,
            plot_container_id=PLOT_CONTAINER_ID,
            scatter_metadata_store_id=SCATTER_METADATA_STORE_ID,
            last_cell_store_id=LAST_CELL_STORE_ID,
            column_handlers=column_handlers,
            default_handler=omega_only_handler,
        )

        selection_lookup = partial(
            resolve_scatter_selection,
            models_data=models_data,
            system_lookup=partial(
                lookup_system_entry,
                data_root=DATA_PATH,  # kept for API compatibility; unused by new helper
                assets_prefix=f"bulk_crystal/{BENCHMARK_NAME}",  # unused by new helper
            ),
        )

        dispersion_renderer = partial(render_dispersion_component, calc_root=CALC_ROOT)

        model_asset_from_scatter(
            scatter_id=SCATTER_GRAPH_ID,
            metadata_store_id=SCATTER_METADATA_STORE_ID,
            asset_container_id=DISPERSION_CONTAINER_ID,
            data_lookup=selection_lookup,
            asset_renderer=dispersion_renderer,
            empty_message="Click on a data point to preview the dispersion + DOS.",
            missing_message="No dispersion plot available for this point.",
        )


def get_app() -> Ti64PhononsApp:
    """
    Construct the Ti64PhononsApp instance.

    Returns
    -------
    Ti64PhononsApp
        Configured application with table + scatter/dispersion panels.
    """
    return Ti64PhononsApp(
        name=BENCHMARK_NAME,
        description=(
            "Accuracy of MLIPs in predicting phonon dispersions and vibrational "
            "thermodynamics for Ti64 alloy."
        ),
        docs_url=DOCS_URL,
        table_path=TABLE_PATH,
        extra_components=[
            dcc.Store(id=LAST_CELL_STORE_ID),
            dcc.Store(id=SCATTER_METADATA_STORE_ID),
            html.Div(
                [
                    html.Div(
                        "Click any cell to view ω_avg (ref vs pred) scatter.",
                        id=PLOT_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                    html.Div(
                        "Click on a data point to preview the dispersion + DOS.",
                        id=DISPERSION_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "24px",
                    "alignItems": "stretch",
                    "flexWrap": "wrap",
                },
            ),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    ti64_app = get_app()
    full_app.layout = ti64_app.layout
    ti64_app.register_callbacks()
    full_app.run(port=8060, debug=True)
