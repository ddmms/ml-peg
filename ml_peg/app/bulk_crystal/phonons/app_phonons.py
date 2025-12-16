"""Run phonon dispersion app."""

from __future__ import annotations

import json
from pathlib import Path

import ase
from dash import Dash, dcc, html
import numpy as np

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    build_classified_parity_scatter,
    build_confusion_heatmap,
    build_violin_distribution,
    figure_from_dict,
    register_scatter_asset_callbacks,
    register_table_plot_callbacks,
)
from ml_peg.calcs import CALCS_ROOT

from .dispersion_assets import render_band_dos_png

DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "phonons"
TABLE_PATH = DATA_PATH / "phonon_metrics_table.json"
SCATTER_PATH = DATA_PATH / "phonon_interactive.json"
BENCHMARK_NAME = "Phonons"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#phonons"
)
CALC_BASE = CALCS_ROOT / "bulk_crystal" / "phonons"

with SCATTER_PATH.open(encoding="utf8") as handle:
    INTERACTIVE_DATA = json.load(handle)

PLOT_CONTAINER_ID = f"{BENCHMARK_NAME}-plot-container"
DISPERSION_CONTAINER_ID = f"{BENCHMARK_NAME}-dispersion-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_NAME}-last-cell"
SCATTER_META_STORE_ID = f"{BENCHMARK_NAME}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_NAME}-scatter"
THZ_TO_K = ase.units._hplanck * 1e12 / ase.units._k


class PhononApp(BaseApp):
    """Phonon benchmark app wiring callbacks and layout."""

    def register_callbacks(self) -> None:
        """Register scatter/dispersion callbacks via shared helpers."""
        calc_root = Path(CALC_BASE)
        models_data = INTERACTIVE_DATA.get("models", {})
        metric_labels = INTERACTIVE_DATA.get("metrics", {})
        label_to_key = {label: key for key, label in metric_labels.items()}
        bz_column = INTERACTIVE_DATA.get("bz_column", "Avg BZ MAE")
        stability_column = INTERACTIVE_DATA.get(
            "stability_column", "Stability Classification (F1)"
        )
        refresh_msg = "Click on a metric to view scatter plots."

        def _metric_points(model_name: str, metric_key: str) -> list[dict]:
            """
            Return scatter points for ``metric_key`` belonging to ``model_name``.

            Parameters
            ----------
            model_name
                Display label of the model in the table.
            metric_key
                Metric identifier corresponding to the column label.

            Returns
            -------
            list[dict]
                Scatter point dictionaries for the requested metric.
            """
            return (
                models_data.get(model_name, {})
                .get("metrics", {})
                .get(metric_key, {})
                .get("points", [])
            )

        def _band_errors(model_name: str) -> dict[str, list[float]]:
            """
            Return Brillouin-zone errors keyed by Materials Project ID.

            Parameters
            ----------
            model_name
                Display label of the model in the table.

            Returns
            -------
            dict[str, list[float]]
                Mapping of MP identifiers to error arrays.
            """
            return models_data.get(model_name, {}).get("band_errors", {})

        def _stability_points(model_name: str) -> list[dict]:
            """
            Return stability scatter data for the given model.

            Parameters
            ----------
            model_name
                Display label of the model in the table.

            Returns
            -------
            list[dict]
                Scatter entries containing reference/prediction values.
            """
            return (
                models_data.get(model_name, {}).get("stability", {}).get("points", [])
            )

        def _system_entry(model_name: str, point_id: str | int) -> dict | None:
            """
            Find the dataset entry (metric or stability) linked to ``point_id``.

            Parameters
            ----------
            model_name
                Display label of the model in the table.
            point_id
                Identifier extracted from the scatter click.

            Returns
            -------
            dict | None
                Matching point metadata or ``None`` when not found.
            """
            if point_id is None:
                return None
            model_entry = models_data.get(model_name, {})
            target = str(point_id)
            for metric_data in model_entry.get("metrics", {}).values():
                for point in metric_data.get("points", []):
                    if str(point.get("id")) == target:
                        return point
            for point in model_entry.get("stability", {}).get("points", []):
                if str(point.get("id")) == target:
                    return point
            return None

        def _handle_metric_column(model_display: str, column_id: str):
            """
            Render the stored scatter figure for non-special columns.

            Parameters
            ----------
            model_display
                Table entry label for the selected model.
            column_id
                Column identifier from the Dash DataTable.

            Returns
            -------
            tuple
                ``(content, meta)`` pair consumed by the shared callback helper.
            """
            metric_key = label_to_key.get(column_id)
            if metric_key is None:
                return None
            figure = figure_from_dict(
                models_data.get(model_display, {}).get("figures", {}).get(metric_key),
                fallback_title=f"No data for {model_display}",
            )
            graph = dcc.Graph(id=SCATTER_GRAPH_ID, figure=figure)
            meta = {"model": model_display, "type": "metric", "metric": metric_key}
            content = html.Div([html.P(refresh_msg), graph])
            return content, meta

        def _handle_bz_column(model_display: str, column_id: str):
            """
            Build the Brillouin-zone violin plot for the selected model.

            Parameters
            ----------
            model_display
                Table entry label for the selected model.
            column_id
                Column identifier (unused but kept for parity with other handlers).

            Returns
            -------
            tuple
                ``(content, meta)`` pair consumed by the shared callback helper.
            """
            band_errors = _band_errors(model_display)
            values: list[float] = []
            labels: list[str] = []
            for mp_id, series in band_errors.items():
                if series is None:
                    continue
                if isinstance(series, (int | float | np.floating | np.integer)):
                    values.append(float(series))
                    labels.append(str(mp_id))
                    continue
                if isinstance(series, np.ndarray):
                    flattened = series.ravel().tolist()
                    values.extend(flattened)
                    labels.extend([str(mp_id)] * len(flattened))
                    continue
                if isinstance(series, (list | tuple)):
                    numeric_vals = [
                        float(val)
                        for val in series
                        if isinstance(val, (int | float | np.floating | np.integer))
                    ]
                    values.extend(numeric_vals)
                    labels.extend([str(mp_id)] * len(numeric_vals))
                    continue
                try:
                    values.append(float(series))
                    labels.append(str(mp_id))
                except (TypeError, ValueError):
                    continue
            fig = build_violin_distribution(
                values,
                labels,
                title=f"{model_display} – Brillouin zone error distribution",
                yaxis_title="|Error| / K",
                hovertemplate="System: %{text}<br>|Error|: %{y:.3f} K<extra></extra>",
            )
            graph = dcc.Graph(id=SCATTER_GRAPH_ID, figure=fig)
            meta = {"model": model_display, "type": "bz"}
            content = html.Div(
                [
                    html.P("Click a violin sample to preview the phonon dispersion."),
                    graph,
                ]
            )
            return content, meta

        def _handle_stability_column(model_display: str, column_id: str):
            """
            Render stability scatter + confusion matrix layout.

            Parameters
            ----------
            model_display
                Table entry label for the selected model.
            column_id
                Column identifier for the clicked stability column.

            Returns
            -------
            tuple
                ``(content, meta)`` pair consumed by the shared callback helper.
            """
            points = _stability_points(model_display)
            scatter = build_classified_parity_scatter(
                points,
                title=f"{model_display} – Stability classification",
                xaxis_title="Reference ω_min / K",
                yaxis_title="Predicted ω_min / K",
                hovertemplate=(
                    "System: %{text}<br>Reference: %{x:.3f} K<br>"
                    "Prediction: %{y:.3f} K<extra></extra>"
                ),
            )
            model_entry = models_data.get(model_display, {})
            confusion_data = model_entry.get("stability", {}).get("confusion") or []
            confusion = build_confusion_heatmap(
                confusion_data,
                x_labels=["DFT stable", "DFT unstable"],
                y_labels=["MLIP stable", "MLIP unstable"],
                title=f"{model_display} – Stability confusion matrix",
                hovertemplate=(
                    "DFT %{x}<br>MLIP %{y}<br>Count: %{z:.0f}<extra></extra>"
                ),
            )
            scatter_graph = dcc.Graph(id=SCATTER_GRAPH_ID, figure=scatter)
            confusion_graph = dcc.Graph(
                id=f"{SCATTER_GRAPH_ID}-confusion", figure=confusion
            )
            meta = {"model": model_display, "type": "stability"}
            content = html.Div(
                [
                    html.P("Click a data point to preview the phonon dispersion plot."),
                    scatter_graph,
                    confusion_graph,
                ]
            )
            return content, meta

        def _lookup_scatter_point(point_data: dict, scatter_meta: dict):
            """
            Resolve the clicked scatter point into its backing metadata entry.

            Parameters
            ----------
            point_data
                Plotly ``point`` dictionary from ``clickData``.
            scatter_meta
                Metadata describing the currently active scatter context.

            Returns
            -------
            dict | None
                Selection context containing ``model`` + ``selection`` keys.
            """
            custom = point_data.get("customdata") or []
            point_id = (
                custom[0]
                if custom
                else point_data.get("text") or point_data.get("label")
            )
            if point_id is None:
                return None
            model_display = scatter_meta.get("model")
            meta_type = scatter_meta.get("type")
            selected = None
            if meta_type == "metric":
                metric_points = _metric_points(
                    model_display, scatter_meta.get("metric")
                )
                selected = next(
                    (entry for entry in metric_points if entry.get("id") == point_id),
                    None,
                )
                if selected is None and isinstance(point_id, int):
                    if 0 <= point_id < len(metric_points):
                        selected = metric_points[point_id]
            elif meta_type == "stability":
                metric_points = _stability_points(model_display)
                selected = next(
                    (entry for entry in metric_points if entry.get("id") == point_id),
                    None,
                )
            elif meta_type == "bz":
                selected = _system_entry(model_display, point_id)
            if not selected:
                return None
            return {"model": model_display, "selection": selected}

        def _render_selected_dispersion(selection_context: dict, scatter_meta: dict):
            """
            Render a Matplotlib dispersion PNG or fallback image for a selection.

            Parameters
            ----------
            selection_context
                Dictionary containing ``model`` and resolved ``selection`` data.
            scatter_meta
                Metadata describing the scatter context (unused, passed for parity).

            Returns
            -------
            dash.html.Div | None
                Component containing the image preview, or ``None`` if missing.
            """
            model_display = selection_context.get("model")
            selected = selection_context.get("selection") or {}
            data_paths = selected.get("data_paths")
            label = selected.get("label") or selected.get("id", "")
            image_src = None
            if data_paths:
                image_src = render_band_dos_png(
                    calc_root=calc_root,
                    paths=data_paths,
                    model_label=model_display,
                    system_label=label,
                    frequency_scale=THZ_TO_K,
                    frequency_unit="K",
                    reference_label="PBE",
                    prediction_label=model_display,
                )
            elif selected.get("image"):
                image_src = f"/{selected['image']}"
            if not image_src:
                return None
            return html.Div(
                [
                    html.H4(label),
                    html.Img(
                        src=image_src,
                        style={"maxWidth": "100%", "border": "1px solid #ccc"},
                    ),
                ]
            )

        column_handlers = {
            bz_column: _handle_bz_column,
            stability_column: _handle_stability_column,
        }

        register_table_plot_callbacks(
            table_id=self.table_id,
            table_data=self.table.data,
            plot_container_id=PLOT_CONTAINER_ID,
            scatter_meta_store_id=SCATTER_META_STORE_ID,
            last_cell_store_id=LAST_CELL_STORE_ID,
            column_handlers=column_handlers,
            default_handler=_handle_metric_column,
            refresh_message=refresh_msg,
        )

        register_scatter_asset_callbacks(
            scatter_id=SCATTER_GRAPH_ID,
            meta_store_id=SCATTER_META_STORE_ID,
            asset_container_id=DISPERSION_CONTAINER_ID,
            data_lookup=_lookup_scatter_point,
            asset_renderer=_render_selected_dispersion,
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
        extra_components=[
            dcc.Store(id=LAST_CELL_STORE_ID),
            dcc.Store(id=SCATTER_META_STORE_ID),
            html.Div(
                [
                    html.Div(
                        "Click on a metric to view scatter plots.",
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
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    phonon_app = get_app()
    full_app.layout = phonon_app.layout
    phonon_app.register_callbacks()
    full_app.run(port=8060, debug=True)
