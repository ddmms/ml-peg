"""Run diamond phonon dispersion app."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import json
from typing import Any

from dash import Dash, dcc, html

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.bulk_crystal.phonons.interactive_helpers import (
    render_dispersion_component,
)
from ml_peg.app.utils.build_callbacks import (
    model_asset_from_scatter,
    scatter_and_assets_from_table,
)
from ml_peg.app.utils.plot_helpers import (
    build_scalar_metric_bar_content,
    build_serialized_scatter_content,
)
from ml_peg.calcs import CALCS_ROOT

BENCHMARK_NAME = "Phonons: Diamond"
BENCHMARK_ID = "phonons_diamond"

DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / BENCHMARK_ID
TABLE_PATH = DATA_PATH / "phonons_diamond_bands_table.json"
SCATTER_PATH = DATA_PATH / "phonons_diamond_bands_interactive.json"
INFO_PATH = DATA_PATH / "info.json"

# Sphinx generates hyphenated anchors from section titles ("Phonons: Diamond").
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html"
    "#phonons-diamond"
)

CALC_BASE = CALCS_ROOT / "bulk_crystal" / BENCHMARK_ID

PLOT_CONTAINER_ID = f"{BENCHMARK_ID}-plot-container"
DISPERSION_CONTAINER_ID = f"{BENCHMARK_ID}-dispersion-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_ID}-last-cell"
SCATTER_METADATA_STORE_ID = f"{BENCHMARK_ID}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_ID}-scatter"


class PhononsDiamondApp(BaseApp):
    """Diamond phonon benchmark app wiring callbacks and layout."""

    def register_callbacks(self) -> None:
        """Register scatter and dispersion callbacks via shared helpers."""
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
                "Hover for point details. Click a point to view the model's "
                "dispersion and DOS."
            ),
        )
        model_order = [row["MLIP"] for row in self.table.data]
        scalar_metric_handler = partial(
            build_scalar_metric_bar_content,
            models_data=models_data,
            label_map=label_to_key,
            model_order=model_order,
            yaxis_titles={
                "gamma": "Absolute error in mean γ",
                "theta_d": "Absolute error in θ_D (K)",
                "kappa": "Absolute error in κ_L (W/m/K)",
            },
            scatter_id=SCATTER_GRAPH_ID,
            instructions=(
                "Each bar is one MLIP. Hover for reference and predicted values. "
                "The model selected in the table is highlighted. Click a bar to "
                "view that model's dispersion and DOS."
            ),
        )
        column_handlers = {
            metric_labels[key]: scalar_metric_handler
            for key in ("gamma", "theta_d", "kappa")
        }

        scatter_and_assets_from_table(
            table_id=self.table_id,
            table_data=self.table.data,
            plot_container_id=PLOT_CONTAINER_ID,
            scatter_metadata_store_id=SCATTER_METADATA_STORE_ID,
            last_cell_store_id=LAST_CELL_STORE_ID,
            column_handlers=column_handlers,
            default_handler=metric_handler,
            scatter_id=SCATTER_GRAPH_ID,
        )

        def model_only_lookup(
            click_data: Mapping[str, Any] | None,
            metadata: Mapping[str, Any],
        ) -> dict[str, Any]:
            """
            Build a selection context for the dispersion preview.

            For this benchmark all plotted data belong to the same system. A bar
            can select another model while a band point retains the table model.

            Parameters
            ----------
            click_data
                Data for the clicked point or bar.
            metadata
                Metadata payload from the scatter callback containing ``model``.

            Returns
            -------
            dict[str, Any]
                Selection context consumed by ``render_dispersion_component``.
            """
            model_name = str(metadata["model"])
            customdata = (click_data or {}).get("customdata", [])
            clicked_model = customdata[0] if customdata else None
            if clicked_model in models_data:
                model_name = str(clicked_model)
            entry = models_data.get(model_name, {})
            return {
                "model": model_name,
                "selection": {
                    "id": "diamond",
                    "label": "Carbon diamond",
                    "data_paths": entry.get("data_paths"),
                    "structure_paths": entry.get("structure_paths"),
                },
            }

        dispersion_renderer = partial(
            render_dispersion_component,
            calc_root=CALC_BASE,
            frequency_scale=1.0,
            frequency_unit="THz",
            reference_label="RSCAN",
        )

        model_asset_from_scatter(
            scatter_id=SCATTER_GRAPH_ID,
            metadata_store_id=SCATTER_METADATA_STORE_ID,
            asset_container_id=DISPERSION_CONTAINER_ID,
            data_lookup=model_only_lookup,
            asset_renderer=dispersion_renderer,
            empty_message="Click on a point or bar to view the dispersion plot.",
            missing_message="No dispersion plot available for this point.",
        )


def get_app() -> PhononsDiamondApp:
    """
    Construct the PhononsDiamondApp instance.

    Returns
    -------
    PhononsDiamondApp
        Configured application with table + scatter/dispersion panels.
    """
    return PhononsDiamondApp(
        name=BENCHMARK_NAME,
        description=(
            "Accuracy of MLIPs in predicting phonon dispersions and thermal "
            "properties for carbon diamond (RSCAN)."
        ),
        docs_url=DOCS_URL,
        table_path=TABLE_PATH,
        extra_components=[
            dcc.Store(id=LAST_CELL_STORE_ID),
            dcc.Store(id=SCATTER_METADATA_STORE_ID),
            html.Div(
                [
                    html.Div(
                        "Click Band MAE for frequency parity, or a thermal metric "
                        "to compare all MLIPs.",
                        id=PLOT_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                    html.Div(
                        "Click on a point or bar to view the dispersion plot.",
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
    phonons_diamond_app = get_app()
    full_app.layout = phonons_diamond_app.layout
    phonons_diamond_app.register_callbacks()
    full_app.run(port=8060, debug=True)
