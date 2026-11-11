"""Stable model ordering and its global table controls."""

from __future__ import annotations

import hashlib
import secrets

from dash import Input, Output, State, callback, ctx, no_update
from dash.dcc import RadioItems, Store
from dash.html import Button, Div, Small, Span


def order_model_rows(rows: list[dict], order: dict | None = None) -> list[dict]:
    """
    Order whole rows without modifying their values or the source list.

    Parameters
    ----------
    rows
        Rows with model names and optional canonical IDs.
    order
        Ordering mode and random seed. Defaults to alphabetical model names.

    Returns
    -------
    list[dict]
        Ordered rows. Random ranks depend only on model identity and seed, so
        filtering and score updates preserve the relative order of other models.
    """
    order = order or {}

    def key(row: dict) -> tuple:
        """
        Return a stable identity-based rank with a deterministic tie-break.

        Parameters
        ----------
        row
            Model row to rank.

        Returns
        -------
        tuple
            Comparable alphabetical or seeded random key.
        """
        name = str(row.get("MLIP", ""))
        identity = str(row.get("id", name))
        alphabetical = (name.casefold(), name, identity)
        if order.get("mode") != "random":
            return alphabetical
        rank = hashlib.sha256(f"{order.get('seed', '')}\0{identity}".encode()).digest()
        return (rank, *alphabetical)

    return sorted(rows, key=key)


def build_model_order_controls() -> Div:
    """
    Build the compact global model ordering controls.

    Returns
    -------
    Div
        Ordering choices, reshuffle button, help text and session state.
    """
    return Div(
        [
            Div(
                [
                    Span("Model order", className="mlpeg-model-order-label"),
                    RadioItems(
                        id="model-order-mode",
                        options=[
                            {"label": "A–Z", "value": "alphabetical"},
                            {"label": "Random", "value": "random"},
                        ],
                        value="alphabetical",
                        inline=True,
                        persistence=True,
                        persistence_type="session",
                    ),
                    Button("Reset A–Z", id="model-order-reshuffle"),
                ],
                className="mlpeg-model-order-options",
            ),
            Small("Alphabetical by default. Column arrows override this order."),
            Store(
                id="model-order-store",
                storage_type="session",
                data={"mode": "alphabetical"},
            ),
        ],
        className="mlpeg-model-order",
    )


def register_model_order_callbacks() -> None:
    """Keep random order stable until the user explicitly asks for another draw."""

    @callback(
        Output("model-order-store", "data"),
        Output("model-order-reshuffle", "children"),
        Input("model-order-mode", "value"),
        Input("model-order-reshuffle", "n_clicks"),
        State("model-order-store", "data"),
    )
    def update_order(mode, clicks, current):
        """
        Apply an explicit ordering choice or retain the current random seed.

        Parameters
        ----------
        mode
            Selected ordering mode.
        clicks
            Number of explicit reset or reshuffle requests.
        current
            Current session ordering state.

        Returns
        -------
        tuple
            Updated state and action button label.
        """
        if mode != "random":
            return {"mode": "alphabetical", "revision": clicks or 0}, "Reset A–Z"
        if (
            current
            and current.get("mode") == "random"
            and ctx.triggered_id != "model-order-reshuffle"
        ):
            return no_update, "Reshuffle"
        return {"mode": "random", "seed": secrets.token_hex(16)}, "Reshuffle"


def register_model_order_reset(table_id: str) -> None:
    """
    Reset column sorting only when the global ordering choice changes.

    Parameters
    ----------
    table_id
        Table whose native column sorting should yield to an explicit choice.
    """

    @callback(
        Output(table_id, "sort_by"),
        Input("model-order-store", "data", allow_optional=True),
        optional=True,
    )
    def reset_column_order(order):
        """
        Let explicit ordering choices override column arrows.

        Parameters
        ----------
        order
            Current ordering state, which triggers a reset when changed.

        Returns
        -------
        list
            Empty native sort specification to reveal the ordered input rows.
        """
        return []
