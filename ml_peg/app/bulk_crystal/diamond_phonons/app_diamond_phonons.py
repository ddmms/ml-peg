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
from ml_peg.app.utils.plot_helpers import build_serialized_scatter_content
from ml_peg.calcs import CALCS_ROOT

BENCHMARK_NAME = "diamond_phonons"

DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / BENCHMARK_NAME
TABLE_PATH = DATA_PATH / "diamond_phonons_bands_table.json"
SCATTER_PATH = DATA_PATH / "diamond_phonons_bands_interactive.json"
INFO_PATH = DATA_PATH / "info.json"

# Sphinx generates hyphenated anchors from section titles ("Diamond phonons").
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html"
    "#diamond-phonons"
)

CALC_BASE = CALCS_ROOT / "bulk_crystal" / BENCHMARK_NAME

PLOT_CONTAINER_ID = f"{BENCHMARK_NAME}-plot-container"
DISPERSION_CONTAINER_ID = f"{BENCHMARK_NAME}-dispersion-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_NAME}-last-cell"
SCATTER_METADATA_STORE_ID = f"{BENCHMARK_NAME}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_NAME}-scatter"


class DiamondPhononApp(BaseApp):
    """Diamond phonon benchmark app wiring callbacks and layout."""

    def register_callbacks(self) -> None:
        """Register scatter and dispersion callbacks via shared helpers."""
        with SCATTER_PATH.open(encoding="utf8") as handle:
            interactive_data = json.load(handle)

        models_data = interactive_data.get("models", {})
        metric_labels = interactive_data.get("metrics", {})
        label_to_key = {label: key for key, label in metric_labels.items()}
        # Thermal metric columns have no per-point scatter data; fall back to
        # the band frequency parity plot so any cell click produces a response.
        for label in self.metrics:
            label_to_key.setdefault(label, "band_mae")

        metric_handler = partial(
            build_serialized_scatter_content,
            models_data=models_data,
            label_map=label_to_key,
            scatter_id=SCATTER_GRAPH_ID,
            instructions=(
                "Click on a metric to view DFT vs predicted frequency scatter plots."
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

        def model_only_lookup(
            click_data: Mapping[str, Any] | None,
            metadata: Mapping[str, Any],
        ) -> dict[str, Any]:
            """
            Build a selection context for the dispersion preview.

            For this benchmark all scatter points belong to the same system,
            so any click shows the model's pre-rendered dispersion.

            Parameters
            ----------
            click_data
                Dash click payload from the scatter plot. Unused.
            metadata
                Metadata payload from the scatter callback; contains ``model``.

            Returns
            -------
            dict[str, Any]
                Selection context consumed by ``render_dispersion_component``.
            """
            _ = click_data
            entry = models_data.get(str(metadata["model"]), {})
            return {
                "model": str(metadata["model"]),
                "selection": {
                    "id": "diamond",
                    "label": "Carbon diamond",
                    "image": entry.get("image"),
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
            empty_message="Click on a scatter point to view the dispersion plot.",
            missing_message="No dispersion plot available for this point.",
        )


def get_app() -> DiamondPhononApp:
    """
    Construct the DiamondPhononApp instance.

    Returns
    -------
    DiamondPhononApp
        Configured application with table + scatter/dispersion panels.
    """
    return DiamondPhononApp(
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
                        "Click on a metric to view DFT vs predicted frequency scatter "
                        "plots.",
                        id=PLOT_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                    html.Div(
                        "Click on a scatter point to view the dispersion plot.",
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
    diamond_phonon_app = get_app()
    full_app.layout = diamond_phonon_app.layout
    diamond_phonon_app.register_callbacks()
    full_app.run(port=8060, debug=True)
