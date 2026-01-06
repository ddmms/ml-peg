"""Components and callbacks for the onboarding walkthrough."""

from __future__ import annotations

from dash import Input, Output, State, callback, ctx, dcc, get_asset_url, html

ONBOARDING_SLIDES: list[dict[str, str]] = [
    {
        "id": "tooltips",
        "title": "Tooltips",
        "description": (
            "Hover over model names and column headers in the tables to get quick "
            "information about each model and test."
        ),
        "video": "onboarding/tooltips.mp4",
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
        "video": "onboarding/interactive-tables-plots.mp4",
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
        "video": "onboarding/weights-thresholds.mp4",
    },
]


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
    base = {
        "position": "fixed",
        "top": "0",
        "left": "0",
        "right": "0",
        "bottom": "0",
        "backgroundColor": "rgba(15, 23, 42, 0.72)",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "zIndex": "1050",
        "padding": "20px",
    }
    base["display"] = "flex" if display else "none"
    return base


def _build_slide(step: int) -> html.Div:
    """
    Create the content block for a given onboarding step.

    Parameters
    ----------
    step : int
        Step index to render content for.

    Returns
    -------
    dash.html.Div
        Div containing the slide caption and video.
    """
    slide = ONBOARDING_SLIDES[step]

    # Video player with actual videos
    video_url = get_asset_url(slide["video"])
    video_container = html.Div(
        [
            html.Video(
                src=video_url,
                autoPlay=True,
                loop=True,
                muted=True,
                # controls=True,
                preload="auto",
                style={
                    "width": "100%",
                    "borderRadius": "8px",
                    "boxShadow": "0 12px 24px rgba(15, 23, 42, 0.3)",
                    "backgroundColor": "#000",
                },
            ),
        ]
    )

    return html.Div(
        [
            html.Div(
                slide["title"].title(),
                style={"fontSize": "20px", "fontWeight": 600, "marginBottom": "8px"},
            ),
            html.P(
                slide["description"],
                style={"marginBottom": "16px", "color": "#475569", "lineHeight": "1.6"},
            ),
            video_container,
        ]
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
                style={
                    "width": "10px",
                    "height": "10px",
                    "borderRadius": "50%",
                    "backgroundColor": "#0d6efd" if active else "#94a3b8",
                    "margin": "0 4px",
                },
                title=slide["title"],
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
        style={
            "position": "fixed",
            "top": "20px",
            "right": "20px",
            "padding": "8px 16px",
            "borderRadius": "6px",
            "border": "1px solid #cbd5e1",
            "background": "white",
            "color": "#475569",
            "cursor": "pointer",
            "fontWeight": 600,
            "fontSize": "14px",
            "zIndex": "1000",
            "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.1)",
            "transition": "all 0.2s ease",
        },
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
                                            "color": "#94a3b8",
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
                                id="onboarding-slide-content",
                                children=_build_slide(0),
                            ),
                            html.Div(
                                id="onboarding-progress-indicator",
                                children=_build_indicator(0),
                                style={"margin": "16px 0"},
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "← Back",
                                        id="onboarding-back-button",
                                        style={
                                            "padding": "10px 20px",
                                            "borderRadius": "6px",
                                            "border": "1px solid #cbd5e1",
                                            "background": "white",
                                            "cursor": "pointer",
                                            "fontWeight": 600,
                                            "transition": "all 0.2s",
                                        },
                                    ),
                                    html.Button(
                                        "Next →",
                                        id="onboarding-next-button",
                                        style={
                                            "padding": "10px 20px",
                                            "borderRadius": "6px",
                                            "border": "none",
                                            "background": "#0d6efd",
                                            "color": "white",
                                            "cursor": "pointer",
                                            "fontWeight": 600,
                                            "transition": "all 0.2s",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "gap": "12px",
                                    "marginTop": "16px",
                                },
                            ),
                        ],
                        style={
                            "background": "white",
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
    """Wire onboarding modal controls and keyboard shortcuts."""
    total = len(ONBOARDING_SLIDES)

    @callback(
        Output("onboarding-step-store", "data"),
        Output("onboarding-state-store", "data"),
        Input("onboarding-next-button", "n_clicks"),
        Input("onboarding-back-button", "n_clicks"),
        Input("onboarding-skip-button", "n_clicks"),
        Input("restart-tutorial-button", "n_clicks"),
        State("onboarding-step-store", "data"),
        State("onboarding-state-store", "data"),
        prevent_initial_call=True,
    )
    def advance(
        next_clicks: int | None,
        back_clicks: int | None,
        skip_clicks: int | None,
        restart_clicks: int | None,
        step_data: dict | None,
        state_data: dict | None,
    ) -> tuple[dict, dict]:
        """
        Handle navigation between slides and manage completion state.

        Parameters
        ----------
        next_clicks, back_clicks, skip_clicks, restart_clicks : int or None
            Button click counts used to determine which control triggered the update.
        step_data : dict or None
            Current onboarding step stored in ``onboarding-step-store``.
        state_data : dict or None
            Completion metadata stored in ``onboarding-state-store``.

        Returns
        -------
        tuple[dict, dict]
            Updated ``step`` payload and completion metadata.
        """
        step = (step_data or {}).get("step", 0)
        state = state_data or {}
        trigger = ctx.triggered_id

        if trigger == "restart-tutorial-button":
            # Reopen tutorial from the beginning
            return {"step": 0}, {"completed": False}
        if trigger == "onboarding-back-button":
            step = max(step - 1, 0)
        elif trigger == "onboarding-next-button":
            if step >= total - 1:
                state["completed"] = True
            else:
                step += 1
        elif trigger == "onboarding-skip-button":
            state["completed"] = True

        return {"step": step}, state

    @callback(
        Output("onboarding-modal-overlay", "style"),
        Output("onboarding-slide-content", "children"),
        Output("onboarding-progress-indicator", "children"),
        Output("onboarding-back-button", "disabled"),
        Output("onboarding-next-button", "children"),
        Input("onboarding-step-store", "data"),
        Input("onboarding-state-store", "data"),
        prevent_initial_call=False,
    )
    def update_modal(
        step_data: dict | None,
        state_data: dict | None,
    ) -> tuple[dict, html.Div, html.Div, bool, str]:
        """
        Update modal visibility and content based on current step.

        Parameters
        ----------
        step_data : dict or None
            Current onboarding step stored in memory.
        state_data : dict or None
            Completion metadata stored in local storage.

        Returns
        -------
        tuple
            Modal style dict, slide content, indicator dots, back-button disabled flag,
            and next-button label.
        """
        state = state_data or {}
        completed = bool(state.get("completed"))
        step = (step_data or {}).get("step", 0)
        step = max(0, min(step, total - 1))

        # Show modal if not completed OR if explicitly restarted
        show_modal = not completed
        modal_style = _overlay_style(show_modal)

        content = _build_slide(step)
        indicator = _build_indicator(step)
        next_label = "Start exploring →" if step == total - 1 else "Next →"
        back_disabled = step == 0

        return modal_style, content, indicator, back_disabled, next_label
