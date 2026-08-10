"""
Header settings panel: theme, colour scheme, benchmark expansion and cache.

Kept in its own file so ``build_app`` stays short. ``build_settings_panel``
builds the gear popover shown in the top-right corner; ``register_settings_callbacks``
wires the theme toggle to ``theme-store`` and the expand-all preference to
``bench-expand-store``.
"""

from __future__ import annotations

from dash import (
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    no_update,
)
from dash.dcc import Checklist, Dropdown, Slider
from dash.exceptions import PreventUpdate
from dash.html import Button, Details, Div, Span, Summary

from ml_peg.app.utils.utils import DEFAULT_COLORMAP

# Table-zoom bounds (percent). Single source for the settings slider and for
# both clamp sites (the apply callback below and the no-flash script in
# run_app.py) so the reachable range and the safety clamps can't diverge.
ZOOM_MIN = 50
ZOOM_MAX = 150


def build_theme_toggle() -> Button:
    """
    Build the light/dark switch shown in the settings panel's Appearance section.

    A pill-shaped toggle whose knob carries a sun (light) or moon (dark). The knob
    position and icon are driven purely by ``html[data-theme]`` in CSS, so it
    renders in the correct state on first paint (the no-flash script in
    ``run_app`` sets ``data-theme`` before Dash boots) with no value pushed back
    into it. Clicking flips ``theme-store`` (see ``register_settings_callbacks``).

    Returns
    -------
    Button
        Header theme toggle.
    """
    return Button(
        Span(className="mlpeg-theme-toggle-knob", **{"aria-hidden": "true"}),
        id="theme-toggle",
        n_clicks=0,
        className="mlpeg-theme-toggle",
        title="Toggle light / dark mode",
        # aria-checked is kept current by the theme-apply callback (CSS alone
        # conveys the knob state visually, but not to screen readers).
        **{"role": "switch", "aria-label": "Toggle dark mode", "aria-checked": "false"},
    )


def build_settings_panel() -> Details:
    """
    Build the settings popover shown in the top-right header.

    A native ``Details`` disclosure (the pattern used app-wide) holding the
    light/dark switch and colour-scheme dropdown (Appearance), plus the
    expand-all-benchmarks preference and the clear-cache button (Behaviour).

    Returns
    -------
    Details
        Settings popover component.
    """
    return Details(
        [
            Summary(
                [
                    # Empty span; the gear glyph is a crisp SVG painted via CSS
                    # mask-image (background: currentColor) so it stays sharp and
                    # follows the theme ink colour.
                    Span(
                        className="mlpeg-settings-gear",
                        **{"aria-hidden": "true"},
                    ),
                    "Settings",
                ],
                className="mlpeg-settings-summary",
                title="App settings",
                # Labelled explicitly: the visible text collapses to just the
                # gear glyph on narrow screens, so screen readers need this.
                **{"aria-label": "App settings"},
            ),
            Div(
                [
                    Div("Appearance", className="mlpeg-settings-section"),
                    Div(
                        [
                            Div(
                                "Theme",
                                className="mlpeg-settings-label",
                            ),
                            build_theme_toggle(),
                        ],
                        className="mlpeg-settings-group mlpeg-settings-row",
                    ),
                    Div(
                        [
                            Div("Colour scheme", className="mlpeg-settings-label"),
                            Dropdown(
                                id="cmap-dropdown",
                                options=[
                                    {
                                        "label": "Viridis (colourblind safe)",
                                        "value": "viridis_r",
                                    },
                                    {
                                        "label": "Blue-Red (colourblind safe)",
                                        "value": "coolwarm",
                                    },
                                    {
                                        "label": "Green-Red",
                                        "value": "RdYlGn_r",
                                    },
                                ],
                                value=DEFAULT_COLORMAP,
                                clearable=False,
                                # No Dash persistence here: the colour scheme is
                                # persisted and restored via ``cmap-store`` + the
                                # sync_cmap callback, which is the single source of
                                # truth. A second persisted copy could diverge on a
                                # partial cache clear.
                                style={"fontSize": "13px", "width": "100%"},
                            ),
                        ],
                        className="mlpeg-settings-group",
                    ),
                    Div(
                        [
                            Div("Table zoom", className="mlpeg-settings-label"),
                            # updatemode="mouseup": the value (and the resize) only
                            # commits when the drag is released — dragging never
                            # continuously re-scales the tables. Scales only the
                            # tables, not the whole page (see theme.css), so the
                            # slider itself never moves under the cursor.
                            Slider(
                                id="zoom-slider",
                                min=ZOOM_MIN,
                                max=ZOOM_MAX,
                                step=10,
                                value=100,
                                marks={
                                    ZOOM_MIN: f"{ZOOM_MIN}%",
                                    100: "100%",
                                    ZOOM_MAX: f"{ZOOM_MAX}%",
                                },
                                tooltip={
                                    "placement": "bottom",
                                    "template": "{value}%",
                                },
                                updatemode="mouseup",
                                className="mlpeg-zoom-slider",
                            ),
                        ],
                        className="mlpeg-settings-group",
                    ),
                    Div(className="mlpeg-settings-divider"),
                    Div("Behaviour", className="mlpeg-settings-section"),
                    Div(
                        Checklist(
                            id="expand-pref-checklist",
                            options=[
                                {
                                    "label": " Expand all benchmarks by default",
                                    "value": "expanded",
                                }
                            ],
                            value=[],
                            className="mlpeg-settings-check",
                        ),
                        className="mlpeg-settings-group",
                    ),
                    Div(
                        Button(
                            "Clear cache",
                            id="clear-storage-button",
                            n_clicks=0,
                            title=(
                                "Clear browser-stored app state (weights, "
                                "thresholds, tutorial progress) and reload. Use "
                                "after an update if the app shows stale data."
                            ),
                            className="mlpeg-settings-clear",
                        ),
                        className="mlpeg-settings-group",
                    ),
                ],
                className="mlpeg-settings-panel",
            ),
        ],
        id="settings-details",
        className="mlpeg-settings",
    )


def register_settings_callbacks() -> None:
    """Register theme sync/apply and expand-all preference sync callbacks."""
    # Flip the theme when the header toggle is clicked. The store may be empty on
    # first load, so fall back to the document attribute (set by the no-flash
    # script in run_app.py) to read the current theme before flipping it. The
    # toggle's visual state is driven from data-theme in CSS, so nothing is
    # pushed back into the button.
    clientside_callback(
        """
        function (nClicks, storedTheme) {
            if (!nClicks) { return window.dash_clientside.no_update; }
            const current = (storedTheme === "dark" || storedTheme === "light")
                ? storedTheme
                : (document.documentElement.getAttribute("data-theme") || "light");
            return current === "dark" ? "light" : "dark";
        }
        """,
        Output("theme-store", "data"),
        Input("theme-toggle", "n_clicks"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )

    # Apply the stored theme to <html data-theme="..."> whenever it changes.
    # NOTE: the no-flash script in run_app.py reads the same value straight
    # from localStorage under the key "theme-store" (dcc.Store persists local
    # stores under their component id) — keep the id and key in sync.
    clientside_callback(
        """
        function (theme) {
            if (theme === "dark" || theme === "light") {
                document.documentElement.setAttribute("data-theme", theme);
            }
            var t = document.documentElement.getAttribute("data-theme") || "light";
            var btn = document.getElementById("theme-toggle");
            if (btn) {
                btn.setAttribute("aria-checked", t === "dark" ? "true" : "false");
            }
            return "";
        }
        """,
        Output("theme-apply-dummy", "children"),
        Input("theme-store", "data"),
    )

    # Apply the stored table zoom as a CSS var (mirrors the theme apply above; the
    # no-flash script sets it before first paint). Only the tables read it (see
    # theme.css), so the sidebar/header/cards stay full size. Clamped as a safety net.
    clientside_callback(
        f"""
        function (zoom) {{
            var v = (typeof zoom === "number" && zoom > 0) ? zoom : 100;
            v = Math.max({ZOOM_MIN}, Math.min({ZOOM_MAX}, v));
            document.documentElement.style.setProperty(
                "--mlpeg-table-zoom", String(v / 100));
            return "";
        }}
        """,
        Output("zoom-apply-dummy", "children"),
        Input("zoom-store", "data"),
    )

    @callback(
        Output("expand-pref-checklist", "value"),
        Output("bench-expand-store", "data", allow_duplicate=True),
        Input("expand-pref-checklist", "value"),
        Input("bench-expand-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_expand_pref(
        checklist_value: list[str] | None,
        stored_pref: str | None,
    ) -> tuple[list[str] | object, str | object]:
        """
        Keep the expand-all preference checkbox and backing store synchronised.

        Parameters
        ----------
        checklist_value
            Current checkbox selection.
        stored_pref
            Persisted preference from ``bench-expand-store`` ("expanded",
            "collapsed" or ``None``).

        Returns
        -------
        tuple[list[str] | object, str | object]
            Checkbox value and store payload; either may be ``dash.no_update``
            depending on which side triggered the sync.
        """
        trigger_id = ctx.triggered_id

        if trigger_id in (None, "bench-expand-store"):
            checked = ["expanded"] if stored_pref == "expanded" else []
            return checked, no_update
        if trigger_id == "expand-pref-checklist":
            expanded = bool(checklist_value) and "expanded" in checklist_value
            return no_update, "expanded" if expanded else "collapsed"
        raise PreventUpdate

    @callback(
        Output("zoom-slider", "value"),
        Output("zoom-store", "data", allow_duplicate=True),
        Input("zoom-slider", "value"),
        Input("zoom-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_zoom(
        slider_value: float | None,
        stored: float | None,
    ) -> tuple[float | object, float | object]:
        """
        Keep the table-zoom slider and backing ``zoom-store`` synchronised.

        Parameters
        ----------
        slider_value
            Current slider percentage.
        stored
            Persisted zoom percentage from ``zoom-store`` (or ``None``).

        Returns
        -------
        tuple[float | object, float | object]
            Slider value and store payload; either may be ``dash.no_update``
            depending on which side triggered the sync.
        """
        trigger_id = ctx.triggered_id

        if trigger_id in (None, "zoom-store"):
            value = stored if isinstance(stored, (int, float)) and stored else 100
            return value, no_update
        if trigger_id == "zoom-slider":
            return no_update, slider_value
        raise PreventUpdate
