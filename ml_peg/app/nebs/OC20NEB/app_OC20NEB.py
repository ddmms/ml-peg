"""Run OC20NEB app."""

from __future__ import annotations

from functools import partial
import json

from dash import Dash, Input, Output, State, callback, dcc, html
from dash.dcc import Loading
from dash.exceptions import PreventUpdate

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.nebs.OC20NEB.interactive_helpers import (
    lookup_system_entry,
    render_geometry_comparison,
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
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "OC20NEB"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#oc20neb"
DATA_PATH = APP_ROOT / "data" / "nebs" / "OC20NEB"
INFO_PATH = DATA_PATH / "info.json"
TABLE_PATH = DATA_PATH / "oc20neb_metrics_table.json"
SCATTER_PATH = DATA_PATH / "oc20neb_interactive.json"
CALC_BASE = CALCS_ROOT / "nebs" / "OC20NEB"

# Component IDs
PLOT_CONTAINER_ID = f"{BENCHMARK_NAME}-plot-container"
PROFILE_CONTAINER_ID = f"{BENCHMARK_NAME}-profile-container"
GEOMETRY_CONTAINER_ID = f"{BENCHMARK_NAME}-geometry-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_NAME}-last-cell"
SCATTER_METADATA_STORE_ID = f"{BENCHMARK_NAME}-scatter-meta"
NEB_PROFILE_STORE_ID = f"{BENCHMARK_NAME}-neb-profile-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_NAME}-scatter"
NEB_PROFILE_GRAPH_ID = f"{BENCHMARK_NAME}-neb-profile"


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

        # ── Table cell → DFT-vs-MLIP scatter plot ────────────────────────────
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

        # ── Scatter point → interactive NEB profile ──────────────────────────
        selection_lookup = partial(
            resolve_scatter_selection,
            models_data=models_data,
            system_lookup=lookup_system_entry,
        )

        profile_renderer = partial(
            render_neb_profile,
            reference_label="RPBE",
            scatter_id=NEB_PROFILE_GRAPH_ID,
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

        # ── NEB profile – store trajectory paths for geometry callback ────────
        # When model_asset_from_scatter fires it stores the full selection_context
        # in NEB_PROFILE_STORE_ID so the geometry callback can read data_paths.
        self._register_neb_profile_store_callback(selection_lookup)

        # ── NEB profile point → side-by-side geometry viewer ─────────────────
        self._register_geometry_callback()

    def _register_neb_profile_store_callback(self, selection_lookup) -> None:
        """
        Persist NEB profile path metadata whenever a scatter point is clicked.

        Stores ``ref_profile``, ``pred_profile``, and ``system_label`` into
        ``NEB_PROFILE_STORE_ID`` so the geometry callback can access trajectory
        paths without re-reading the JSON.

        Parameters
        ----------
        selection_lookup
            Callable that resolves a clicked scatter point and scatter metadata
            into a selection context dict containing ``data_paths`` and a label.
        """

        @callback(
            Output(NEB_PROFILE_STORE_ID, "data"),
            Input(SCATTER_GRAPH_ID, "clickData"),
            State(SCATTER_METADATA_STORE_ID, "data"),
            prevent_initial_call=True,
        )
        def _store_neb_paths(click_data, scatter_metadata):
            """
            Store trajectory paths for the clicked scatter point.

            Parameters
            ----------
            click_data
                Plotly ``clickData`` event dict from the scatter graph.
            scatter_metadata
                Stored metadata identifying the active model and metric context.

            Returns
            -------
            dict
                Mapping of ``ref_profile``, ``pred_profile``, and
                ``system_label`` written to ``NEB_PROFILE_STORE_ID``.
            """
            if not click_data or not scatter_metadata:
                raise PreventUpdate
            points = click_data.get("points")
            if not points:
                raise PreventUpdate

            point_data = points[0]
            selection_context = selection_lookup(point_data, scatter_metadata)
            if not selection_context:
                raise PreventUpdate

            selected = selection_context.get("selection") or {}
            data_paths = selected.get("data_paths") or {}
            label = selected.get("label") or selected.get("reaction", "")

            return {
                "ref_profile": data_paths.get("ref_profile"),
                "pred_profile": data_paths.get("pred_profile"),
                "system_label": label,
            }

    def _register_geometry_callback(self) -> None:
        """
        Render side-by-side DFT and MLIP geometries on NEB profile point click.

        Reads the clicked ``pointNumber`` (NEB image index) together with the
        trajectory paths stored in ``NEB_PROFILE_STORE_ID`` and delegates
        rendering to ``render_geometry_comparison``.
        """

        @callback(
            Output(GEOMETRY_CONTAINER_ID, "children"),
            Input(NEB_PROFILE_GRAPH_ID, "clickData"),
            State(NEB_PROFILE_STORE_ID, "data"),
            prevent_initial_call=True,
        )
        def _show_geometry(click_data, neb_profile_meta):
            """
            Render side-by-side geometry viewers on NEB profile point click.

            Parameters
            ----------
            click_data
                Plotly ``clickData`` event dict from the NEB profile graph.
                ``pointNumber`` is used as the NEB image index.
            neb_profile_meta
                Stored dict containing ``ref_profile``, ``pred_profile``, and
                ``system_label`` written by ``_store_neb_paths``.

            Returns
            -------
            dash.html.Div
                Side-by-side WEAS geometry viewers, or an informational message
                when no data is available for the selected image.
            """
            if not click_data or not neb_profile_meta:
                raise PreventUpdate

            points = click_data.get("points")
            if not points:
                raise PreventUpdate

            # pointNumber gives the NEB image index (same for both traces)
            image_index = points[0].get("pointNumber", 0)

            click_context = {
                "ref_profile": neb_profile_meta.get("ref_profile"),
                "pred_profile": neb_profile_meta.get("pred_profile"),
                "system_label": neb_profile_meta.get("system_label", ""),
                "image_index": image_index,
            }

            result = render_geometry_comparison(click_context)
            if result is None:
                return html.Div("No geometry data available for this image.")
            return result


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
        description="Accuracy of MLIPs in predicting NEB profiles.",
        docs_url=DOCS_URL,
        table_path=TABLE_PATH,
        extra_components=[
            # ── Stores ────────────────────────────────────────────────────────
            dcc.Store(id=LAST_CELL_STORE_ID),
            dcc.Store(id=SCATTER_METADATA_STORE_ID),
            dcc.Store(id=NEB_PROFILE_STORE_ID),
            # ── Row 1: DFT-vs-MLIP scatter | NEB profile ─────────────────────
            html.Div(
                [
                    html.Div(
                        "Click on a metric to view scatter plots.",
                        id=PLOT_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                    Loading(
                        html.Div(
                            "Click on a data point to preview the NEB profile.",
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
                    "marginBottom": "24px",
                },
            ),
            # ── Row 2: side-by-side geometry viewer ───────────────────────────
            Loading(
                html.Div(
                    "Click on a point in NEB profile to view DFT and MLIP geometries",
                    id=GEOMETRY_CONTAINER_ID,
                    style={"minHeight": "80px"},
                ),
                type="circle",
            ),
        ],
        info_path=INFO_PATH,
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
