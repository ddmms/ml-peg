"""Run Ti64 phonon dispersion + DOS + free-energy app."""

from __future__ import annotations

from functools import partial
import json

from dash import Dash, dcc, html

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.bulk_crystal.phonons.interactive_helpers import (
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
from ml_peg.calcs import CALCS_ROOT

BENCHMARK_NAME = "Phonons: Ti64"
BENCHMARK_ID = "phonons_ti64"

DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / BENCHMARK_ID
TABLE_PATH = DATA_PATH / "phonons_ti64_metrics_table.json"
SCATTER_PATH = DATA_PATH / "phonons_ti64_interactive.json"
INFO_PATH = DATA_PATH / "info.json"

# Sphinx generates hyphenated anchors from section titles ("Phonons: Ti64").
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html"
    "#phonons-ti64"
)

CALC_BASE = CALCS_ROOT / "bulk_crystal" / BENCHMARK_ID

PLOT_CONTAINER_ID = f"{BENCHMARK_ID}-plot-container"
DISPERSION_CONTAINER_ID = f"{BENCHMARK_ID}-dispersion-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_ID}-last-cell"
SCATTER_METADATA_STORE_ID = f"{BENCHMARK_ID}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_ID}-scatter"


class PhononsTi64App(BaseApp):
    """Ti64 phonons benchmark app wiring callbacks and layout."""

    def register_callbacks(self) -> None:
        """Register scatter/dispersion callbacks via shared helpers."""
        with SCATTER_PATH.open(encoding="utf8") as handle:
            interactive_data = json.load(handle)

        models_data = interactive_data.get("models", {})
        metric_labels = interactive_data.get("metrics", {})
        label_to_key = {label: key for key, label in metric_labels.items()}

        metric_handler = partial(
            build_serialized_scatter_content,
            models_data=models_data,
            label_map=label_to_key,
            scatter_id=SCATTER_GRAPH_ID,
            instructions=(
                "Each point is one Ti64 case. Hover for values. Click a point "
                "to view its dispersion and DOS."
            ),
        )

        scatter_and_assets_from_table(
            table_id=self.table_id,
            table_data=self.table.data,
            plot_container_id=PLOT_CONTAINER_ID,
            scatter_metadata_store_id=SCATTER_METADATA_STORE_ID,
            last_cell_store_id=LAST_CELL_STORE_ID,
            column_handlers={},
            default_handler=metric_handler,
            scatter_id=SCATTER_GRAPH_ID,
        )

        selection_lookup = partial(
            resolve_scatter_selection,
            models_data=models_data,
            system_lookup=lookup_system_entry,
        )
        dispersion_renderer = partial(
            render_dispersion_component,
            calc_root=CALC_BASE,
            frequency_scale=1.0,
            frequency_unit="THz",
            reference_label="PBE",
        )

        model_asset_from_scatter(
            scatter_id=SCATTER_GRAPH_ID,
            metadata_store_id=SCATTER_METADATA_STORE_ID,
            asset_container_id=DISPERSION_CONTAINER_ID,
            data_lookup=selection_lookup,
            asset_renderer=dispersion_renderer,
            empty_message="Click on a data point to preview the dispersion + DOS.",
            missing_message="No dispersion plot available for this point.",
        )


def get_app() -> PhononsTi64App:
    """
    Construct the PhononsTi64App instance.

    Returns
    -------
    PhononsTi64App
        Configured application with table + scatter/dispersion panels.
    """
    return PhononsTi64App(
        name=BENCHMARK_NAME,
        description=(
            "Accuracy of MLIPs in predicting phonon dispersions and vibrational "
            "thermodynamics for Ti64 alloy phases."
        ),
        docs_url=DOCS_URL,
        table_path=TABLE_PATH,
        extra_components=[
            dcc.Store(id=LAST_CELL_STORE_ID),
            dcc.Store(id=SCATTER_METADATA_STORE_ID),
            html.Div(
                [
                    html.Div(
                        "Click a metric to compare its per-case predictions with PBE.",
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
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    phonons_ti64_app = get_app()
    full_app.layout = phonons_ti64_app.layout
    phonons_ti64_app.register_callbacks()
    full_app.run(port=8060, debug=True)
