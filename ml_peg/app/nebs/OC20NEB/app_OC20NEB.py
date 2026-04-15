"""Run OC20NEB app."""

from __future__ import annotations

from functools import partial
import json

from dash import Dash, dcc, html
from dash.dcc import Loading

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.nebs.OC20NEB.interactive_helpers import (
    lookup_system_entry,
    render_neb_profile,
)
from ml_peg.app.utils.build_callbacks import (
    model_asset_from_scatter,
    scatter_and_assets_from_table,
)
from ml_peg.app.utils.plot_helpers import (
    build_serialized_scatter_content,
    resolve_scatter_selection,
)
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "OC20NEB"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#oc20neb"
DATA_PATH = APP_ROOT / "data" / "nebs" / "OC20NEB"
TABLE_PATH = DATA_PATH / "oc20neb_metrics_table.json"
SCATTER_PATH = DATA_PATH / "oc20neb_interactive.json"
CALC_BASE = CALCS_ROOT / "nebs" / "OC20NEB"

PLOT_CONTAINER_ID = f"{BENCHMARK_NAME}-plot-container"
PROFILE_CONTAINER_ID = f"{BENCHMARK_NAME}-profiile-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_NAME}-last-cell"
SCATTER_METADATA_STORE_ID = f"{BENCHMARK_NAME}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_NAME}-scatter"


class OC20NEBApp(BaseApp):
    """OC20NEB benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        with SCATTER_PATH.open(encoding="utf8") as handle:
            interactive_data = json.load(handle)

        models_data = interactive_data.get("models", {})
        metric_labels = interactive_data.get("metrics", {})
        label_to_key = {label: key for key, label in metric_labels.items()}

        refresh_msg = "Click on a metric to view scatter plots."

        metric_handler = partial(
            build_serialized_scatter_content,
            models_data=models_data,
            label_map=label_to_key,
            scatter_id=SCATTER_GRAPH_ID,
            instructions=refresh_msg,
        )

        scatter_and_assets_from_table(
            table_id=self.table_id,
            table_data=self.table.data,
            plot_container_id=PLOT_CONTAINER_ID,
            scatter_metadata_store_id=SCATTER_METADATA_STORE_ID,
            last_cell_store_id=LAST_CELL_STORE_ID,
            column_handlers={},
            default_handler=metric_handler,
        )

        selection_lookup = partial(
            resolve_scatter_selection,
            models_data=models_data,
            system_lookup=lookup_system_entry,
        )
        profile_renderer = partial(
            render_neb_profile,
            reference_label="RPBE",
        )

        model_asset_from_scatter(
            scatter_id=SCATTER_GRAPH_ID,
            metadata_store_id=SCATTER_METADATA_STORE_ID,
            asset_container_id=PROFILE_CONTAINER_ID,
            data_lookup=selection_lookup,
            asset_renderer=profile_renderer,
            empty_message="Click on a data point to preview the NEB profile.",
            missing_message="No profile plot available for this point.",
        )


def get_app() -> OC20NEBApp:
    """
    Get OC20NEB benchmark app layout and callback registration.

    Returns
    -------
    OC20NEBApp
        Benchmark layout and callback registration.
    """
    return OC20NEBApp(
        name=BENCHMARK_NAME,
        description=("Accuracy of MLIPs in predicting NEB profiles."),
        docs_url=DOCS_URL,
        table_path=TABLE_PATH,
        # Dash Stores persist the last clicked cell and the scatter metadata that
        # identifies the selected system + model so the OC20NEB callback can
        # look up the correct asset paths when rendering neb profile plots
        extra_components=[
            dcc.Store(id=LAST_CELL_STORE_ID),
            dcc.Store(id=SCATTER_METADATA_STORE_ID),
            html.Div(
                [
                    html.Div(
                        "Click on a metric to view scatter plots.",
                        id=PLOT_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                    Loading(
                        html.Div(
                            "Click on a scatter point to view the dispersion plot.",
                            id=PROFILE_CONTAINER_ID,
                            style={"flex": "1", "minWidth": 0},
                        ),
                        type="circle",
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
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    oc20neb_app = get_app()
    full_app.layout = oc20neb_app.layout
    oc20neb_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
