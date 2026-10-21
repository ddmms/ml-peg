"""Central controls backed by the existing category and benchmark weight stores."""

from __future__ import annotations

from dash.dash_table import DataTable
from dash.html import Button, Details, Div, Label, Summary

from ml_peg.app.utils.build_components import build_weight_input
from ml_peg.app.utils.register_callbacks import register_weight_callbacks


def _weight_row(table: DataTable, column: str, reset_id: str) -> Div:
    """
    Build a labelled input connected to its existing table store.

    Parameters
    ----------
    table
        Summary table owning the weight store.
    column
        Weighted column identifier.
    reset_id
        Reset button mounted alongside this input.

    Returns
    -------
    Div
        Label and numeric weight input.
    """
    prefix = f"global-{table.id}-{column}"
    defaults = getattr(table, "weights", None) or {}
    register_weight_callbacks(prefix, table.id, column, defaults, reset_id)
    control = build_weight_input(f"{prefix}-input", defaults.get(column, 1.0))
    control.style.update(width="76px", flexShrink="0")
    return Div(
        [
            Label(column.removesuffix(" Score"), htmlFor=f"{prefix}-input"),
            control,
        ],
        className="mlpeg-global-weight-row",
    )


def build_global_weights(summary_table: DataTable, category_views: dict) -> Details:
    """
    Build the central category and benchmark weight editor.

    Parameters
    ----------
    summary_table
        Overall table whose weights determine category contributions.
    category_views
        Category metadata containing each benchmark summary table.

    Returns
    -------
    Details
        Collapsible controls sharing stores with the existing table inputs.
    """
    reset_id = "global-category-weights-reset"
    categories = []
    summary_columns = {col["id"] for col in summary_table.columns}
    for title, view in sorted(category_views.items()):
        column = f"{title} Score"
        if column not in summary_columns:
            continue
        table = view["summary_table"]
        benchmark_reset = f"global-{table.id}-reset"
        benchmarks = [
            _weight_row(table, col["id"], benchmark_reset)
            for col in table.columns
            if col["id"] not in {"MLIP", "Score", "id", "link"}
        ]
        categories.append(
            Details(
                [
                    Summary(_weight_row(summary_table, column, reset_id)),
                    Div(
                        [
                            *benchmarks,
                            Button(
                                "Reset benchmarks",
                                id=benchmark_reset,
                                n_clicks=0,
                                className="mlpeg-scoring-reset",
                            ),
                        ],
                        className="mlpeg-global-weight-benchmarks",
                    ),
                ],
                className="mlpeg-global-weight-category",
            )
        )
    return Details(
        [
            Summary("All weights"),
            Div(
                [
                    Div(
                        "Change weights, then press Enter or click away.",
                        className="mlpeg-global-weights-hint",
                    ),
                    *categories,
                    Button(
                        "Reset category weights",
                        id=reset_id,
                        n_clicks=0,
                        className="mlpeg-scoring-reset",
                    ),
                ],
                className="mlpeg-global-weights-body",
            ),
        ],
        id="global-weights",
        className="mlpeg-global-weights",
    )
