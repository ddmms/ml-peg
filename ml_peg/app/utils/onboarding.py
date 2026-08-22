"""Components and callbacks for the onboarding walkthrough."""

from __future__ import annotations

import json

from dash import Input, Output, State, clientside_callback, dcc, get_asset_url, html

# Step-dot colours, shared by the Python-built initial dots and the clientside
# step renderer (f-stringed into its JS). Theme tokens, so the dots follow dark
# mode; React and the DOM both accept var() in inline styles.
DOT_ACTIVE_COLOUR = "var(--mlpeg-accent)"
DOT_IDLE_COLOUR = "var(--mlpeg-muted)"

ONBOARDING_SLIDES: list[dict[str, str]] = [
    {
        "id": "tooltips",
        "title": "Tooltips",
        "description": (
            "Hover over model names and column headers in the tables to get quick "
            "information about each model and test."
        ),
        "video": "ui/onboarding/tooltips.mp4",
    },
    {
        "id": "plots",
        "title": "Interactive tables and plots",
        "description": (
            "Test tables and plots are interactive! Click table cells to show the "
            "data from which each result is calculated, then click points in the "
            "plots to dive into the datapoint either showing where the data comes "
            "from (e.g. phonon dispersion) or a structure visualisation."
        ),
        "video": "ui/onboarding/interactive-tables-plots.mp4",
    },
    {
        "id": "weights-thresholds",
        "title": "Weights and normalisation thresholds",
        "description": (
            "Use the weights and thresholds controls to customise how models are "
            "scored and ranked, based on your needs. Adjust the weights to "
            "prioritise certain tests or metrics, and set 'Good' and 'Bad' "
            "thresholds to alter the linear normalisation of scores."
        ),
        "video": "ui/onboarding/weights-thresholds.mp4",
    },
]


# Modal overlay chrome, single-sourced: `_overlay_style` bakes the initial
# (hidden) state into the layout and the clientside step renderer re-emits the
# same dict (JSON-injected) with only `display` toggled. The scrim stays a fixed
# dark colour in both themes — a dark scrim reads correctly over light and dark
# surfaces alike.
_OVERLAY_BASE: dict[str, str] = {
    "position": "fixed",
    "top": "0",
    "left": "0",
    "right": "0",
    "bottom": "0",
    "backgroundColor": "rgba(15, 23, 42, 0.72)",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "zIndex": "2000",  # Above loading overlays (1200/1400).
    "padding": "20px",
}


def _overlay_style(display: bool = False) -> dict[str, str]:
    """
    Return modal overlay styles toggled by ``display``.

    Parameters
    ----------
    display : bool, optional
        Whether the overlay should be visible.

    Returns
    -------
    dict[str, str]
        CSS styles applied to the overlay container.
    """
    return _OVERLAY_BASE | {"display": "flex" if display else "none"}


def _build_caption(step: int, active: bool = True) -> html.Div:
    """
    Build the title + description block for a step.

    Captions and videos are split into two toggled tracks so the navigation
    arrows can overlay the video without the caption height (which varies per
    step) shifting them. All captions are mounted once and toggled client-side by
    ``data-idx`` (see :func:`register_onboarding_callbacks`).

    Parameters
    ----------
    step : int
        Step index to render.
    active : bool, optional
        Whether this caption starts visible (only the first step does).

    Returns
    -------
    dash.html.Div
        Caption block tagged for clientside toggling.
    """
    slide = ONBOARDING_SLIDES[step]
    return html.Div(
        [
            html.Div(
                slide["title"].title(),
                style={"fontSize": "20px", "fontWeight": 600, "marginBottom": "8px"},
            ),
            html.P(
                slide["description"],
                style={
                    "marginBottom": "16px",
                    "color": "var(--mlpeg-ink-2)",
                    "lineHeight": "1.6",
                },
            ),
        ],
        className="onboarding-caption",
        style={"display": "block" if active else "none"},
        **{"data-idx": str(step)},
    )


def _build_video(step: int, active: bool = True) -> html.Div:
    """
    Build the video block for a step.

    All videos are mounted once and toggled client-side by ``data-idx``, so
    stepping never re-fetches a video -- each ``<video>`` loads once and switching
    is instant. The render callback plays the active slide's video and pauses the
    others; hence no ``autoPlay`` here. The videos ship with ``preload="none"``
    so an unopened tutorial costs no bytes -- the render callback flips the
    active slide's video to ``preload="auto"`` just before playing it.

    Parameters
    ----------
    step : int
        Step index to render.
    active : bool, optional
        Whether this video starts visible (only the first step does).

    Returns
    -------
    dash.html.Div
        Video block tagged for clientside toggling.
    """
    slide = ONBOARDING_SLIDES[step]
    video_url = get_asset_url(slide["video"])
    return html.Div(
        html.Video(
            src=video_url,
            loop=True,
            muted=True,
            preload="none",
            style={
                "width": "100%",
                "display": "block",
                "borderRadius": "8px",
                "backgroundColor": "#000",
            },
        ),
        className="onboarding-video",
        style={"display": "block" if active else "none"},
        **{"data-idx": str(step)},
    )


def _build_indicator(step: int) -> html.Div:
    """
    Render dot indicators that reflect the active onboarding step.

    Parameters
    ----------
    step : int
        Currently active slide index.

    Returns
    -------
    dash.html.Div
        Div containing the dot elements.
    """
    dots = []
    for idx, slide in enumerate(ONBOARDING_SLIDES):
        active = idx == step
        dots.append(
            html.Div(
                className="onboarding-dot",
                style={
                    "width": "10px",
                    "height": "10px",
                    "borderRadius": "50%",
                    "backgroundColor": (
                        DOT_ACTIVE_COLOUR if active else DOT_IDLE_COLOUR
                    ),
                    "margin": "0 4px",
                },
                title=slide["title"],
                **{"data-idx": str(idx)},
            )
        )
    return html.Div(dots, style={"display": "flex", "justifyContent": "center"})


def build_tutorial_button() -> html.Button:
    """
    Create a "Restart Tour" button for the app header.

    Returns
    -------
    html.Button
        Button that reopens the onboarding modal when clicked.
    """
    return html.Button(
        "Tutorial",
        id="restart-tutorial-button",
        title="Restart the interactive tutorial",
        # Positioning is handled by the header-controls container in build_app so
        # this button can sit alongside the clear-cache button. Visual styling is
        # fully overridden by the .mlpeg-header-actions > button CSS.
        style={"cursor": "pointer"},
    )


def build_onboarding_modal() -> html.Div:
    """
    Create onboarding modal shell with stores for state management.

    Returns
    -------
    html.Div
        Wrapper containing stores and the modal overlay.
    """
    return html.Div(
        [
            # Stores for managing state
            dcc.Store(
                id="onboarding-step-store",
                storage_type="memory",
                data={"step": 0},
            ),
            dcc.Store(
                id="onboarding-state-store",
                storage_type="local",
                data={},
            ),
            # Modal overlay
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Welcome to ML-PEG",
                                        style={"fontSize": "24px", "fontWeight": 700},
                                    ),
                                    html.Button(
                                        "✕",
                                        id="onboarding-skip-button",
                                        title=(
                                            "Close tutorial (you can reopen it anytime)"
                                        ),
                                        style={
                                            "background": "transparent",
                                            "border": "none",
                                            "color": "var(--mlpeg-ink-3)",
                                            "cursor": "pointer",
                                            "fontWeight": 600,
                                            "fontSize": "24px",
                                            "lineHeight": "1",
                                            "padding": "0",
                                            "width": "30px",
                                            "height": "30px",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "12px",
                                },
                            ),
                            html.Div(
                                id="onboarding-caption-content",
                                # Captions and videos are two toggled tracks so the
                                # overlay arrows sit on the video, not below the
                                # variable-height caption. All mounted once; the
                                # render callback toggles by data-idx.
                                children=[
                                    _build_caption(i, active=(i == 0))
                                    for i in range(len(ONBOARDING_SLIDES))
                                ],
                            ),
                            html.Div(
                                [
                                    *[
                                        _build_video(i, active=(i == 0))
                                        for i in range(len(ONBOARDING_SLIDES))
                                    ],
                                    # Transparent centre catcher: click the video to
                                    # advance. Sits below the arrows in stacking
                                    # order so edge clicks hit the arrows, not this.
                                    html.Div(
                                        id="onboarding-advance",
                                        className="onboarding-advance",
                                        n_clicks=0,
                                        title="Next",
                                    ),
                                    html.Button(
                                        "‹",
                                        id="onboarding-nav-prev",
                                        n_clicks=0,
                                        className="onboarding-nav onboarding-nav--prev",
                                        **{"aria-label": "Previous step"},
                                    ),
                                    html.Button(
                                        "›",
                                        id="onboarding-nav-next",
                                        n_clicks=0,
                                        className="onboarding-nav onboarding-nav--next",
                                        **{"aria-label": "Next step"},
                                    ),
                                ],
                                id="onboarding-video-stage",
                                className="onboarding-video-stage",
                            ),
                            html.Div(
                                id="onboarding-progress-indicator",
                                children=_build_indicator(0),
                                style={"margin": "18px 0 0"},
                            ),
                        ],
                        style={
                            "background": "var(--mlpeg-surface)",
                            "borderRadius": "12px",
                            "width": "min(680px, 90vw)",
                            "maxHeight": "90vh",
                            "overflowY": "auto",
                            "padding": "28px",
                            "boxShadow": "0 25px 50px rgba(15, 23, 42, 0.4)",
                            "position": "relative",
                        },
                    ),
                ],
                id="onboarding-modal-overlay",
                style=_overlay_style(False),
            ),
        ]
    )


def register_onboarding_callbacks() -> None:
    """
    Wire the onboarding modal entirely client-side.

    Both navigation (arrows / click-to-advance / Skip / Restart -> step) and
    rendering (show/hide the active caption + video, recolour the dots, play the
    active video) run in the browser, so stepping is instant: no server round-trip
    and no video re-download.
    """
    total = len(ONBOARDING_SLIDES)

    # Advance the step / completion state from whichever control was clicked:
    # the ‹ / › arrows, the video click-catcher (advance), Skip (✕) or Restart.
    clientside_callback(
        f"""
        function(skipN, restartN, prevN, nextN, advanceN, stepData, stateData) {{
            const total = {total};
            const ctx = window.dash_clientside.callback_context;
            const trig = (ctx && ctx.triggered && ctx.triggered.length)
                ? ctx.triggered[0].prop_id.split('.')[0] : null;
            let step = (stepData && stepData.step) || 0;
            let state = Object.assign({{}}, stateData || {{}});
            if (trig === 'restart-tutorial-button') {{
                return [{{step: 0}}, {{completed: false}}];
            }}
            if (trig === 'onboarding-nav-prev') {{
                step = Math.max(step - 1, 0);
            }} else if (trig === 'onboarding-nav-next'
                       || trig === 'onboarding-advance') {{
                if (step >= total - 1) {{ state.completed = true; }}
                else {{ step += 1; }}
            }} else if (trig === 'onboarding-skip-button') {{
                state.completed = true;
            }}
            return [{{step: step}}, state];
        }}
        """,
        Output("onboarding-step-store", "data"),
        Output("onboarding-state-store", "data"),
        Input("onboarding-skip-button", "n_clicks"),
        Input("restart-tutorial-button", "n_clicks"),
        Input("onboarding-nav-prev", "n_clicks"),
        Input("onboarding-nav-next", "n_clicks"),
        Input("onboarding-advance", "n_clicks"),
        State("onboarding-step-store", "data"),
        State("onboarding-state-store", "data"),
        prevent_initial_call=True,
    )

    # Reflect step/completion in the DOM: modal visibility is the declared output;
    # the pre-mounted captions, videos, dots and nav arrows are toggled directly
    # (they are static, so React never clobbers these changes). The ‹ arrow hides
    # on the first step; the › arrow becomes ✓ on the last so click/next finishes.
    clientside_callback(
        f"""
        function(stepData, stateData) {{
            const total = {total};
            const state = stateData || {{}};
            const completed = !!state.completed;
            let step = (stepData && stepData.step) || 0;
            step = Math.max(0, Math.min(step, total - 1));
            document.querySelectorAll('.onboarding-caption').forEach(function(el) {{
                const on = parseInt(el.getAttribute('data-idx'), 10) === step;
                el.style.display = on ? 'block' : 'none';
            }});
            document.querySelectorAll('.onboarding-video').forEach(function(el) {{
                const on = parseInt(el.getAttribute('data-idx'), 10) === step;
                el.style.display = on ? 'block' : 'none';
                const v = el.querySelector('video');
                if (v) {{
                    if (on && !completed) {{
                        v.preload = 'auto';
                        v.play().catch(function() {{}});
                    }} else {{ v.pause(); }}
                }}
            }});
            document.querySelectorAll('.onboarding-dot').forEach(function(el) {{
                const on = parseInt(el.getAttribute('data-idx'), 10) === step;
                el.style.backgroundColor = on
                    ? '{DOT_ACTIVE_COLOUR}' : '{DOT_IDLE_COLOUR}';
            }});
            const prev = document.getElementById('onboarding-nav-prev');
            if (prev) {{ prev.style.visibility = step === 0 ? 'hidden' : 'visible'; }}
            const next = document.getElementById('onboarding-nav-next');
            if (next) {{
                const last = step === total - 1;
                next.textContent = last ? '✓' : '›';
                next.title = last ? 'Finish' : 'Next';
            }}
            const overlay = Object.assign({{}}, {json.dumps(_OVERLAY_BASE)});
            overlay.display = completed ? 'none' : 'flex';
            return overlay;
        }}
        """,
        Output("onboarding-modal-overlay", "style"),
        Input("onboarding-step-store", "data"),
        Input("onboarding-state-store", "data"),
        prevent_initial_call=False,
    )
