"""Run phonon dispersion app."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path

import ase
from dash import Dash, dcc, html
from dash.dcc import Loading

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.bulk_crystal.phonons.interactive_helpers import (
    build_bz_violin_content,
    lookup_system_entry,
    render_dispersion_component,
)
from ml_peg.app.utils.build_callbacks import (
    model_asset_from_scatter,
    scatter_and_assets_from_table,
)
from ml_peg.app.utils.plot_helpers import (
    build_classification_panel,
    build_serialized_scatter_content,
    resolve_scatter_selection,
)
from ml_peg.calcs import CALCS_ROOT

DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "phonons"
TABLE_PATH = DATA_PATH / "phonon_metrics_table.json"
SCATTER_PATH = DATA_PATH / "phonon_interactive.json"
BENCHMARK_NAME = "Phonons"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#phonons"
)
CALC_BASE = CALCS_ROOT / "bulk_crystal" / "phonons"

PLOT_CONTAINER_ID = f"{BENCHMARK_NAME}-plot-container"
DISPERSION_CONTAINER_ID = f"{BENCHMARK_NAME}-dispersion-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_NAME}-last-cell"
SCATTER_METADATA_STORE_ID = f"{BENCHMARK_NAME}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_NAME}-scatter"
THZ_TO_K = ase.units._hplanck * 1e12 / ase.units._k


class PhononApp(BaseApp):
    """Phonon benchmark app wiring callbacks and layout."""

    def register_callbacks(self) -> None:
        """Register scatter/dispersion callbacks via shared helpers."""
        with SCATTER_PATH.open(encoding="utf8") as handle:
            interactive_data = json.load(handle)

        calc_root = Path(CALC_BASE)
        models_data = interactive_data.get("models", {})
        metric_labels = interactive_data.get("metrics", {})
        label_to_key = {label: key for key, label in metric_labels.items()}
        bz_column = interactive_data.get("bz_column", "Avg BZ MAE")
        stability_column = interactive_data.get(
            "stability_column", "Stability Classification (F1)"
        )
        refresh_msg = "Click on a metric to view scatter plots."

        metric_handler = partial(
            build_serialized_scatter_content,
            models_data=models_data,
            label_map=label_to_key,
            scatter_id=SCATTER_GRAPH_ID,
            instructions=refresh_msg,
        )
        bz_handler = partial(
            build_bz_violin_content,
            models_data=models_data,
            scatter_id=SCATTER_GRAPH_ID,
            instructions="Click a violin sample to preview the phonon dispersion.",
            yaxis_title="|Error| / K",
            hovertemplate="System: %{text}<br>|Error|: %{y:.3f} K<extra></extra>",
        )
        stability_handler = partial(
            build_classification_panel,
            models_data=models_data,
            scatter_id=SCATTER_GRAPH_ID,
            confusion_id=f"{SCATTER_GRAPH_ID}-confusion",
            instructions="Click a data point to preview the phonon dispersion plot.",
            scatter_hovertemplate=(
                "System: %{text}<br>Reference: %{x:.3f} K<br>"
                "Prediction: %{y:.3f} K<extra></extra>"
            ),
            xaxis_title="Reference ω_min / K",
            yaxis_title="Predicted ω_min / K",
            confusion_axes=(
                ["DFT stable", "DFT unstable"],
                ["MLIP stable", "MLIP unstable"],
            ),
            classification_key="stability",
        )
        column_handlers = {bz_column: bz_handler, stability_column: stability_handler}

        scatter_and_assets_from_table(
            table_id=self.table_id,
            table_data=self.table.data,
            plot_container_id=PLOT_CONTAINER_ID,
            scatter_metadata_store_id=SCATTER_METADATA_STORE_ID,
            last_cell_store_id=LAST_CELL_STORE_ID,
            column_handlers=column_handlers,
            default_handler=metric_handler,
        )

        selection_lookup = partial(
            resolve_scatter_selection,
            models_data=models_data,
            system_lookup=lookup_system_entry,
        )
        dispersion_renderer = partial(
            render_dispersion_component,
            calc_root=calc_root,
            frequency_scale=THZ_TO_K,
            frequency_unit="K",
            reference_label="PBE",
        )

        model_asset_from_scatter(
            scatter_id=SCATTER_GRAPH_ID,
            metadata_store_id=SCATTER_METADATA_STORE_ID,
            asset_container_id=DISPERSION_CONTAINER_ID,
            data_lookup=selection_lookup,
            asset_renderer=dispersion_renderer,
            empty_message="Click on a data point to preview the phonon dispersion.",
            missing_message="No dispersion plot available for this point.",
        )


def get_app() -> PhononApp:
    """
    Construct the PhononApp instance.

    Returns
    -------
    PhononApp
        Configured application with table + scatter/dispersion panels.
    """
    return PhononApp(
        name=BENCHMARK_NAME,
        description=(
            "Accuracy of MLIPs in predicting phonon dispersions and vibrational "
            "thermodynamics for bulk crystals."
        ),
        docs_url=DOCS_URL,
        table_path=TABLE_PATH,
        # Dash Stores persist the last clicked cell and the scatter metadata that
        # identifies the selected system + model so the dispersion callback can
        # look up the correct asset paths when rendering dispersion plots
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
                            id=DISPERSION_CONTAINER_ID,
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
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    phonon_app = get_app()
    full_app.layout = phonon_app.layout
    phonon_app.register_callbacks()
    full_app.run(port=8060, debug=True)
