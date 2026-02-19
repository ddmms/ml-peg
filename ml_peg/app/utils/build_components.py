"""Utility functions for building app components."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import time

from dash import html
from dash.dash_table import DataTable
from dash.dcc import Checklist, Store
from dash.dcc import Input as DCC_Input
from dash.development.base_component import Component
from dash.html import H2, H3, Br, Button, Details, Div, Label, Summary
import yaml

from ml_peg.analysis.utils.utils import Thresholds
from ml_peg.app.utils.register_callbacks import (
    register_category_table_callbacks,
    register_normalization_callbacks,
    register_summary_table_callbacks,
    register_weight_callbacks,
)
from ml_peg.app.utils.utils import calculate_column_widths


def grid_template_from_widths(
    widths: dict[str, int],
    column_order: list[str],
) -> str:
    """
    Compose a CSS grid template string from column widths.

    Parameters
    ----------
    widths
        Mapping of column names to pixel widths.
    column_order
        Ordered metric column names to render between the MLIP and Score columns.

    Returns
    -------
    str
        CSS grid template definition using `minmax` tracks.
    """
    tracks: list[tuple[str, int]] = [("MLIP", widths["MLIP"])]
    tracks.extend((col, widths[col]) for col in column_order)
    tracks.append(("Score", widths["Score"]))

    template_parts: list[str] = []
    for _, width in tracks:
        min_px = max(width, 40)
        weight = max(width / 10, 1)
        template_parts.append(f"minmax({int(min_px)}px, {weight:.3f}fr)")
    return " ".join(template_parts)


def build_weight_input(
    input_id: str,
    default_value: float | None,
    show_label: bool = False,
) -> Div:
    """
    Build numeric input for a metric weight.

    Parameters
    ----------
    input_id
        ID for text box input component.
    default_value
        Default value for the text box input.
    show_label
        Whether to show the "Weight:" label (only for first column).

    Returns
    -------
    Div
        Div wrapping the input box.
    """
    wrapper_style: dict[str, str] = {
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "boxSizing": "border-box",
        "border": "1px solid transparent",
        "position": "relative",
    }
    wrapper_style.update(
        {
            "width": "100%",
            "minWidth": "0",
            "maxWidth": "100%",
            "height": "100%",
        }
    )

    children = []
    if show_label:
        children.append(
            Label(
                "Weight:",
                style={
                    "fontSize": "13px",
                    "color": "#6c757d",
                    "textAlign": "right",
                    "position": "absolute",
                    "right": "calc(50% + 38px)",
                },
            )
        )

    children.append(
        DCC_Input(
            id=input_id,
            type="number",
            value=default_value,
            step=0.01,
            debounce=True,
            style={
                "width": "60px",
                "fontSize": "12px",
                "padding": "2px 4px",
                "border": "1px solid #6c757d",
                "borderRadius": "3px",
                "textAlign": "center",
            },
        )
    )

    return Div(children, style=wrapper_style)


def build_weight_components(
    header: str,
    table: DataTable,
    *,
    use_thresholds: bool = False,
    column_widths: dict[str, int] | None = None,
    thresholds: Thresholds | None = None,
) -> Div:
    """
    Build weight sliders, text boxes and reset button.

    Parameters
    ----------
    header
        Header for above sliders.
    table
        DataTable to build weight components for.
    use_thresholds
        Whether this table also exposes normalization thresholds. When True,
        weight callbacks will reuse the raw-data store and normalization store to
        recompute Scores consistently.
    column_widths
        Optional mapping of table column IDs to pixel widths used to align the
        inputs with the rendered table.
    thresholds
        Threshold metadata used when ``use_thresholds`` is ``True``.

    Returns
    -------
    Div
        Div containing header, weight sliders, text boxes and reset button.
    """
    if use_thresholds and thresholds is None:
        raise ValueError(
            "Threshold metadata must be provided when use_thresholds=True."
        )
    # Identify metric columns (exclude reserved columns)
    reserved = {"MLIP", "Score", "id"}
    columns = [col["id"] for col in table.columns if col.get("id") not in reserved]

    if not columns:
        return Div()

    input_ids = [f"{table.id}-{col}" for col in columns]

    # Set default weights
    weights = table.weights if table.weights else {}
    for column in columns:
        weights.setdefault(column, 1.0)

    widths = calculate_column_widths(columns, column_widths)
    grid_template = grid_template_from_widths(widths, columns)

    weight_inputs = [
        build_weight_input(
            input_id=f"{input_id}-input",
            default_value=weights[column],
            show_label=(i == 0),
        )
        for i, (column, input_id) in enumerate(zip(columns, input_ids, strict=True))
    ]

    container = Div(
        [
            Div(
                [
                    Div(
                        header,
                        style={
                            "fontWeight": "bold",
                            "fontSize": "13px",
                            "padding": "2px 4px",
                            "color": "#212529",
                            "whiteSpace": "nowrap",
                            "boxSizing": "border-box",
                            "border": "1px solid transparent",
                        },
                    ),
                    Button(
                        "Reset",
                        id=f"{table.id}-reset-button",
                        n_clicks=0,
                        style={
                            "fontSize": "11px",
                            "padding": "4px 8px",
                            "marginTop": "0px",
                            "marginLeft": "4px",
                            "backgroundColor": "#6c757d",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "3px",
                            "width": "fit-content",
                            "cursor": "pointer",
                        },
                    ),
                    Div(
                        "Press Enter or click away to apply new weights or thresholds",
                        style={
                            "fontSize": "11px",
                            "color": "#6c757d",
                            "fontStyle": "italic",
                            "marginTop": "2px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-start",
                    "gap": "4px",
                    "boxSizing": "border-box",
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "border": "1px solid transparent",  # #dee2e6 or transparent
                },
            ),
            *weight_inputs,
            Div(
                "",
                style={
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            ),
            Div(
                "",
                style={
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": grid_template,
            "alignItems": "start",
            "columnGap": "0px",
            "rowGap": "4px",
            "marginTop": "-5px",
            "padding": "2px 4px",
            "backgroundColor": "#f8f9fa",
            "border": "1px solid transparent"
            if header == "Metric Weights"
            else "1px solid #dee2e6",
            "borderRadius": "6px",
            "width": "100%",
            "minWidth": "0",
            "boxSizing": "border-box",
        },
    )

    layout = [
        Br(),
        container,
        Store(
            id=f"{table.id}-weight-store",
            storage_type="session",
            data=weights,
        ),
    ]

    model_levels = getattr(table, "model_levels_of_theory", None)
    metric_levels = getattr(table, "metric_levels_of_theory", None)
    model_configs = getattr(table, "model_configs", None)

    # Callbacks to update table scores when table weight dicts change
    if table.id != "summary-table":
        register_category_table_callbacks(
            table_id=table.id,
            use_thresholds=use_thresholds,
            model_levels=model_levels,
            metric_levels=metric_levels,
            model_configs=model_configs,
        )
    else:
        register_summary_table_callbacks(
            model_levels=model_levels,
            metric_levels=metric_levels,
            model_configs=model_configs,
        )

    # Callbacks to sync sliders, text boxes, and stored table weights
    for column, input_id in zip(columns, input_ids, strict=True):
        register_weight_callbacks(
            input_id=input_id,
            table_id=table.id,
            column=column,
            default_weights=getattr(table, "weights", None),
        )

    return Div(layout)


def build_faqs() -> Div:
    """
    Build FAQ section with collapsible dropdowns from YAML file.

    Returns
    -------
    Div
        Styled FAQ section with questions as dropdown titles and answers inside.
    """
    # Load FAQs from YAML file
    faqs_path = Path(__file__).parent / "faqs.yml"

    try:
        with open(faqs_path, encoding="utf8") as f:
            faqs_data = yaml.safe_load(f)
    except FileNotFoundError:
        return Div(
            "FAQs file not found",
            style={
                "color": "#dc3545",
                "padding": "10px",
                "fontStyle": "italic",
            },
        )

    if not faqs_data or not isinstance(faqs_data, list):
        return Div("No FAQs available")

    # Build FAQ dropdowns
    faq_components = []

    for faq in faqs_data:
        question = faq.get("question", "")
        answer = faq.get("answer", "")
        docs_url = faq.get("docs_url")

        if not question or not answer:
            continue

        # Build answer content with optional docs link
        answer_content = [answer]
        if docs_url:
            answer_content.append(
                Div(
                    html.A(
                        "View documentation →",
                        href=docs_url,
                        target="_blank",
                        style={
                            "color": "#0d6efd",
                            "textDecoration": "none",
                            "fontWeight": "600",
                            "fontSize": "13px",
                        },
                    ),
                    style={"marginTop": "8px"},
                )
            )

        faq_components.append(
            Details(
                [
                    Summary(
                        question,
                        style={
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "padding": "10px",
                            "backgroundColor": "#f8f9fa",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "4px",
                            "marginBottom": "8px",
                        },
                    ),
                    Div(
                        answer_content,
                        style={
                            "padding": "10px 15px",
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #dee2e6",
                            "borderTop": "none",
                            "borderRadius": "0 0 4px 4px",
                            "marginTop": "-8px",
                            "marginBottom": "8px",
                        },
                    ),
                ],
                style={
                    "marginBottom": "8px",
                },
            )
        )

    return Div(
        [
            H2(
                "Frequently Asked Questions",
                style={"color": "black", "marginTop": "30px"},
            ),
            Div(faq_components),
        ]
    )


def build_footer() -> html.Footer:
    """
    Build shared footer with author, copyright, and repository link.

    Returns
    -------
    html.Footer
        Styled footer component rendered at the bottom of each tab.
    """
    copyright_first_year = "2025"
    current_year = str(time.localtime().tm_year)
    copyright_owners = metadata.metadata("ml-peg")["author"]

    copyright_year_string = (
        current_year
        if current_year == copyright_first_year
        else f"{copyright_first_year}-{current_year}"
    )
    copyright_txt = (
        f"Copyright © {copyright_year_string}, {copyright_owners}. All rights reserved"
    )

    return html.Footer(
        [
            Div(
                [
                    html.Span(copyright_txt, style={"fontWeight": "800"}),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "flexWrap": "wrap",
                    "gap": "6px",
                },
            ),
            Div(
                html.A(
                    "GPL-3.0 license",
                    href="https://github.com/ddmms/ml-peg/blob/main/LICENSE",
                    target="_blank",
                ),
                style={"marginTop": "4px", "fontWeight": "800"},
            ),
            Div(
                html.A(
                    [
                        html.Img(
                            src=(
                                "https://github.githubassets.com/images/modules/logos_page/"
                                "GitHub-Mark.png"
                            ),
                            alt="GitHub",
                            width="16",
                            height="16",
                            style={
                                "marginRight": "6px",
                                "verticalAlign": "middle",
                                "display": "block",
                            },
                        ),
                        html.Span("View on GitHub"),
                    ],
                    href="https://github.com/ddmms/ml-peg",
                    target="_blank",
                    style={
                        "color": "#0d6efd",
                        "textDecoration": "none",
                        "fontWeight": "800",
                        "display": "inline-flex",
                        "alignItems": "center",
                    },
                ),
                style={"marginTop": "6px"},
            ),
        ],
        style={
            "marginTop": "24px",
            "padding": "14px 12px",
            "color": "#343a40",
            "fontSize": "12px",
            "textAlign": "center",
            "borderTop": "1px solid #dee2e6",
            "background": "#f8f9fa",
            "borderRadius": "6px",
        },
    )


def build_test_layout(
    name: str,
    description: str,
    table: DataTable,
    extra_components: list[Component] | None = None,
    docs_url: str | None = None,
    column_widths: dict[str, int] | None = None,
    thresholds: Thresholds | None = None,
) -> Div:
    """
    Build app layout for a test.

    Parameters
    ----------
    name
        Name of test.
    description
        Description of test.
    table
        Dash Table with metric results. Can include a `weights` attribute to be used by
        `build_weight_components`.
    extra_components
        List of Dash Components to include after the metrics table.
    docs_url
        URL to online documentation. Default is None.
    column_widths
        Optional column-width mapping inferred from analysis output. Used to align
        threshold controls beneath the table columns when available.
    thresholds
        Optional normalization metadata (metric -> (good, bad, unit)) supplied via the
        analysis pipeline. When provided, inline threshold controls are rendered
        automatically.

    Returns
    -------
    Div
        Layout for test layout.
    """
    layout_contents = [
        H2(name, style={"color": "black"}),
        H3(description),
    ]

    layout_contents.extend(
        [
            Details(
                [
                    Summary(
                        "Click for more information",
                        style={
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "padding": "5px",
                        },
                    ),
                    Label(
                        [html.A("Online documentation", href=docs_url, target="_blank")]
                    ),
                ],
                style={
                    # "border": "1px solid #ddd",
                    "padding": "10px",
                    # "borderRadius": "5px",
                },
            ),
            Br(),
        ]
    )

    layout_contents.append(Div(table))
    layout_contents.append(
        Store(
            id=f"{table.id}-computed-store",
            storage_type="session",
            data=table.data,
        )
    )

    # Inline normalization thresholds when metadata is supplied
    if thresholds is not None:
        reserved = {"MLIP", "Score", "id"}
        metric_columns = [
            col["id"] for col in table.columns if col.get("id") not in reserved
        ]
        layout_contents.append(
            Store(
                id=f"{table.id}-raw-data-store",
                storage_type="session",
                data=table.data,
            )
        )
        layout_contents.append(
            Store(
                id=f"{table.id}-raw-tooltip-store",
                storage_type="session",
                data=table.tooltip_header,
            )
        )
        threshold_controls = build_threshold_inputs(
            table_columns=metric_columns,
            thresholds=thresholds,
            table_id=table.id,
            column_widths=column_widths,
        )

    # Add metric-weight controls for every benchmark table
    metric_weights = build_weight_components(
        header="Metric Weights",
        table=table,
        use_thresholds=True,
        column_widths=column_widths,
        thresholds=thresholds,
    )
    if threshold_controls and metric_weights:
        # Combine threshold and weight panels in a single card while trimming the extra
        # <Br/> injected at the top of the weight component so the boundary box hugs
        # both controls.
        # The first child of the weight component is always the spacer <Br/> returned by
        # build_weight_components. Drop it from the weights so the metric weights box
        # hugs the threshold box.
        weight_children = metric_weights.children
        weight_children = weight_children[1:]
        compact_weights = Div(weight_children)

        # Insert a single spacer before the combined card so its top aligns with the
        # elements above (e.g. the table). The thresholds + weights content then sit
        # within the shared box.
        layout_contents.append(Br())
        layout_contents.append(
            Div(
                [
                    Div(threshold_controls, style={"marginBottom": "0px"}),
                    Div(compact_weights, style={"marginTop": "0"}),
                ],
                style={
                    "backgroundColor": "#f8f9fa",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "6px",
                    "padding": "0px 0px 0px 0px",  # top right bottom left
                    "marginTop": "-5px",
                    "boxSizing": "border-box",
                    "width": "100%",
                },
            )
        )
    elif threshold_controls:
        layout_contents.append(threshold_controls)
    elif metric_weights:
        layout_contents.append(metric_weights)

    layout_contents.append(
        Store(
            id="summary-table-scores-store",
            storage_type="session",
        ),
    )

    if extra_components:
        layout_contents.extend(extra_components)

    return Div(layout_contents)


def build_threshold_inputs(
    table_columns: list[str],
    thresholds: Thresholds,
    table_id: str,
    column_widths: dict[str, int] | None = None,
) -> Div:
    """
    Build inline Good/Bad threshold inputs aligned to the table columns.

    Parameters
    ----------
    table_columns : list[str]
        Ordered metric column names in the table.
    thresholds : Thresholds
        Default (good, bad) threshold ranges keyed by metric column.
    table_id : str
        Identifier prefix for threshold inputs and related controls.
    column_widths : dict[str, int] or None
        Optional pixel widths used to align the grid with the table columns.

    Returns
    -------
    Div
        Container with threshold inputs and associated controls.
    """
    widths = calculate_column_widths(table_columns, column_widths)
    grid_template = grid_template_from_widths(widths, table_columns)

    container_style = {
        "display": "grid",
        "gridTemplateColumns": grid_template,
        "alignItems": "start",
        "justifyItems": "center",
        "columnGap": "0px",
        "rowGap": "0px",
        "marginTop": "0px",
        "padding": "2px 2px",
        "backgroundColor": "#f8f9fa",
        "border": "1px solid transparent",
        "borderRadius": "5px",
        "width": "100%",
        "minWidth": "0",
        "boxSizing": "border-box",
    }

    cells: list[Div] = []
    default_thresholds: Thresholds = {}

    cells.append(
        Div(
            [
                Div(
                    "Thresholds",
                    style={
                        "fontWeight": "bold",
                        "fontSize": "13px",
                        "padding": "2px 4px",
                        "whiteSpace": "nowrap",
                        "boxSizing": "border-box",
                        "color": "#212529",
                        "border": "1px solid transparent",
                    },
                ),
                Button(
                    "Reset",
                    id=f"{table_id}-reset-thresholds-button",
                    n_clicks=0,
                    style={
                        "fontSize": "11px",
                        "padding": "4px 8px",
                        "marginTop": "0px",
                        "marginLeft": "4px",
                        "backgroundColor": "#6c757d",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "3px",
                        "width": "fit-content",
                        "cursor": "pointer",
                    },
                ),
                # Toggle to view normalized metric values in the table
                Checklist(
                    id=f"{table_id}-normalized-toggle",
                    options=[{"label": "Show normalised scores", "value": "norm"}],
                    value=[],
                    style={"marginTop": "6px", "fontSize": "11px"},
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "inline-flex", "alignItems": "center"},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "flex-start",
                "justifyContent": "center",
                "padding": "1px 2px",
                "justifySelf": "start",
                "width": "100%",
                "minWidth": "0",
                "maxWidth": "100%",
                "boxSizing": "border-box",
                "border": "1px solid transparent",
            },
        )
    )

    for metric in table_columns:
        bounds = thresholds.get(metric)
        good_val, bad_val, unit_label = bounds["good"], bounds["bad"], bounds["unit"]
        default_thresholds[metric] = {
            "good": good_val,
            "bad": bad_val,
            "unit": unit_label,
        }

        first_metric = metric == table_columns[0]

        good_children = [
            Label(
                "Good:" if first_metric else "",
                style={
                    "fontSize": "13px",
                    "color": "lightseagreen",
                    "textAlign": "right",
                    "position": "absolute",
                    "right": "calc(50% + 40px)",
                },
            ),
            DCC_Input(
                id=f"{table_id}-{metric}-good-threshold",
                type="number",
                value=good_val,
                step=0.001,
                debounce=True,
                style={
                    "width": "60px",
                    "fontSize": "12px",
                    "padding": "2px 4px",
                    "border": "1px solid lightseagreen",
                    "borderRadius": "3px",
                    "margin": "0 auto",
                    "display": "block",
                },
            ),
        ]
        if unit_label:
            good_children.append(
                html.Span(
                    f"[{unit_label}]",
                    style={
                        "fontSize": "12px",
                        "color": "#6c757d",
                        "position": "absolute",
                        "left": "calc(50% + 40px)",
                        "top": "50%",
                        "transform": "translateY(-50%)",
                        "whiteSpace": "nowrap",
                    },
                )
            )

        bad_children = [
            Label(
                "Bad:" if first_metric else "",
                style={
                    "fontSize": "13px",
                    "color": "#dc3545",
                    "textAlign": "right",
                    "position": "absolute",
                    "right": "calc(50% + 40px)",
                    "top": "50%",
                    "transform": "translateY(-50%)",
                    "whiteSpace": "nowrap",
                },
            ),
            DCC_Input(
                id=f"{table_id}-{metric}-bad-threshold",
                type="number",
                value=bad_val,
                step=0.001,
                debounce=True,
                style={
                    "width": "60px",
                    "fontSize": "12px",
                    "padding": "2px 4px",
                    "border": "1px solid #dc3545",
                    "borderRadius": "3px",
                    "margin": "0 auto",
                    "display": "block",
                },
            ),
        ]
        if unit_label:
            bad_children.append(
                html.Span(
                    f"[{unit_label}]",
                    style={
                        "fontSize": "12px",
                        "color": "#6c757d",
                        "position": "absolute",
                        "left": "calc(50% + 40px)",
                        "top": "50%",
                        "transform": "translateY(-50%)",
                        "whiteSpace": "nowrap",
                    },
                )
            )

        cells.append(
            Div(
                [
                    Div(
                        good_children,
                        style={
                            "position": "relative",
                            "width": "100%",
                            "padding": "4px 32px",
                            "boxSizing": "border-box",
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "marginBottom": "2px",
                        },
                    ),
                    Div(
                        bad_children,
                        style={
                            "position": "relative",
                            "width": "100%",
                            "padding": "4px 32px",
                            "boxSizing": "border-box",
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "padding": "2px 0",
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "height": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            )
        )

    # Score
    cells.append(
        Div(
            "",
            style={
                "width": "100%",
                "minWidth": "0",
                "maxWidth": "100%",
                "boxSizing": "border-box",
                "border": "1px solid transparent",
            },
        )
    )

    store = Store(
        id=f"{table_id}-thresholds-store",
        storage_type="session",
        data=default_thresholds,
    )

    # Register callbacks for these metrics, pass default_thresholds for reset
    register_normalization_callbacks(
        table_id,
        table_columns,
        default_thresholds,
        register_toggle=False,
    )

    return Div(
        [
            Div(cells, id=f"{table_id}-threshold-grid", style=container_style),
            store,
        ]
    )
