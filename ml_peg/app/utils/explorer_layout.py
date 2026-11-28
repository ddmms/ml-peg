"""
Layout of the ML-PEG Model Explorer page.

Answers "which model should I use, and where will it stop working?" with seven
panels: model finder, report card, head-to-head, where models fail, selected
structure, who fails alike, and the coverage rose.
"""

from __future__ import annotations

from dash import dcc, html
from dash.development.base_component import Component
import plotly.graph_objects as go

from ml_peg.analysis.explorer.build_explorer_data import OUTLIER_FACTOR
from ml_peg.app.utils.explorer_data import (
    explorer_data_ready,
    load_benchmark_scores,
    load_error_matrix,
    load_hardness,
    load_similarity,
    model_taxonomy,
)
from ml_peg.app.utils.explorer_figures import (
    build_coverage_rose_figure,
    build_finder_figure,
    build_head_to_head_figure,
    build_report_profile_figure,
    build_similarity_heatmap_figure,
    build_structure_fails_figure,
    finder_legend,
    graph_style,
    lot_class_legend,
    render_fails_list,
    render_finder_picks,
    render_h2h_summary,
    render_report_summary,
    render_report_table,
)
from ml_peg.app.utils.explorer_ids import (
    COVERAGE_GRANULARITY,
    COVERAGE_GRANULARITY_OPTIONS,
    COVERAGE_GRAPH,
    FINDER_CATEGORIES,
    FINDER_ELEMENTS,
    FINDER_EXPERIMENTS,
    FINDER_GRAPH,
    FINDER_PICKS,
    FINDER_PRESET,
    FINDER_TOLERANCE,
    FINDER_TOLERANCE_OPTIONS,
    H2H_DETAIL_CATEGORY,
    H2H_DETAIL_GRAPH,
    H2H_GRAPH,
    H2H_MODEL_A,
    H2H_MODEL_B,
    H2H_SUMMARY,
    REPORT_MODEL,
    REPORT_PROFILE,
    REPORT_SUMMARY,
    REPORT_TABLE,
    SIMILARITY_GRAPH,
    STRUCT_FAILS_CATEGORIES,
    STRUCT_FAILS_GRAPH,
    STRUCT_FAILS_LIST,
    STRUCT_FAILS_MODEL,
    STRUCT_FAILS_SORT,
    STRUCT_FAILS_SORT_OPTIONS,
    STRUCT_TABLE,
    STRUCT_VIEWER,
)
from ml_peg.app.utils.explorer_scoring import (
    FAILS_PER_BENCHMARK,
    FINDER_BENCH_WEIGHT,
    H2H_TIE_BAND,
    MIN_FINDER_BENCH,
    cached_finder_rankings,
    cached_structure_fails_ranking,
    categories,
    default_h2h_models,
    element_options,
    h2h_area_options,
    h2h_pair_stats,
    model_options,
    offered_categories,
    report_default_model,
    representative_models,
)

SCORING_GUIDE_URL = (
    "https://ddmms.github.io/ml-peg/developer_guide/scoring_and_normalisation.html"
)

# Panel card class: the same card as the home/category summary tables, so the
_PANEL_CLASS = "mlpeg-summary-card"

_EXPLORER_JUMP_NAV: list[tuple[str, str]] = [
    ("Find a model", "explorer-sec-finder"),
    ("Report card", "explorer-sec-report"),
    ("Head-to-head", "explorer-sec-h2h"),
    ("Where models fail", "explorer-sec-fails"),
    ("Selected structure", "explorer-sec-detail"),
    ("Who fails alike", "explorer-sec-alike"),
    ("Coverage", "explorer-sec-coverage"),
]

_CONTROL_FONT = {"fontSize": "13px"}
_RADIO_INPUT_STYLE = {"marginRight": "6px"}
# Space below a standalone control row, matching .mlpeg-explorer-controls
_ROW_GAP = {"marginBottom": "var(--sp-4)"}


def build_explorer_layout() -> html.Div:
    """
    Return the Explorer page.

    Shows a guidance message instead if the startup data build failed.

    Returns
    -------
    html.Div
        The page content for the ``/explorer`` route.
    """
    if not explorer_data_ready():
        return html.Div(
            [
                html.H1("ML-PEG Model Explorer"),
                html.P(
                    "The explorer data could not be built from the shipped benchmarks; "
                    "the server log records the error. Fix the data and restart the "
                    "app."
                ),
            ]
        )
    return build_explorer_body()


def build_explorer_body() -> html.Div:
    """
    Build the full Explorer page (assumes the data is computed and cached).

    Returns
    -------
    html.Div
        The intro and the seven panels plus the methods panel, or a guidance message
        if no benchmark data is shipped.
    """
    rows = load_hardness()
    benchmark_scores = load_benchmark_scores()
    if not rows and not benchmark_scores:
        return html.Div(
            [
                html.H1("ML-PEG Model Explorer"),
                html.P(
                    "No benchmark data is available to explore yet. Add benchmark "
                    "results under ml_peg/app/data and reload."
                ),
            ]
        )

    models = load_similarity().get("models", [])
    # Default the comparison to representatives of two *different* families -- a model
    # vs its own +D3 twin is a near-degenerate first view.
    reps = representative_models(models) or models
    default_a, default_b = default_h2h_models(reps, load_error_matrix())
    category_names = categories(rows)
    variant_meta = model_taxonomy()["variants"]
    h2h_options = [
        {
            "label": f"{m}  ·  {variant_meta.get(m, {}).get('training_lot') or '?'}",
            "value": m,
        }
        for m in models
    ]

    return html.Div(
        [
            _build_intro(len(rows), len(models), len(benchmark_scores)),
            _build_jump_nav(),
            html.Div(
                _build_finder_panel(
                    sorted(offered_categories()), element_options(rows)
                ),
                id="explorer-sec-finder",
            ),
            html.Div(_build_report_panel(), id="explorer-sec-report"),
            html.Div(
                _build_h2h_panel(h2h_options, default_a, default_b),
                id="explorer-sec-h2h",
            ),
            html.Div(
                _build_structure_fails_panel(category_names, model_options()),
                id="explorer-sec-fails",
            ),
            html.Div(_build_detail_panel(), id="explorer-sec-detail"),
            html.Div(_build_similarity_panel(), id="explorer-sec-alike"),
            html.Div(_build_coverage_panel(), id="explorer-sec-coverage"),
            _build_references_panel(),
        ],
        className="mlpeg-explorer",
    )


def _chip(text: str) -> html.Div:
    """
    Render a small info chip matching the app's styling.

    Parameters
    ----------
    text
        Chip label.

    Returns
    -------
    html.Div
        The chip element.
    """
    return html.Div(text, className="mlpeg-chip")


def _labelled_control(label: str, control: Component) -> html.Div:
    """
    Wrap a labelled control (dropdown or radio group) in the app's control styling.

    Parameters
    ----------
    label
        Control label.
    control
        The input component.

    Returns
    -------
    html.Div
        The labelled control.
    """
    return html.Div(
        [
            html.Label(label, htmlFor=control.id, className="mlpeg-control-label"),
            control,
        ]
    )


def _radio_items(
    component_id: str, options: list[dict[str, str]], value: str
) -> dcc.RadioItems:
    """
    Build an inline radio group with the Explorer's control typography.

    Parameters
    ----------
    component_id
        The component id.
    options
        The ``{"label", "value"}`` options.
    value
        The initially selected value.

    Returns
    -------
    dcc.RadioItems
        The radio group.
    """
    return dcc.RadioItems(
        id=component_id,
        options=options,
        value=value,
        inline=True,
        className="mlpeg-explorer-radio",
        style=_CONTROL_FONT,
        inputStyle=_RADIO_INPUT_STYLE,
    )


def _graph(component_id: str, figure: go.Figure, **style: str) -> dcc.Graph:
    """
    Build a responsive graph whose wrapper height follows its figure.

    Parameters
    ----------
    component_id
        The graph's component id.
    figure
        The initial figure.
    **style
        Extra style entries (e.g. ``display="none"``).

    Returns
    -------
    dcc.Graph
        The graph.
    """
    return dcc.Graph(
        id=component_id,
        figure=figure,
        responsive=True,
        style={**graph_style(figure), **style},
        config={"displaylogo": False},
    )


def _a11y_graph(graph: dcc.Graph, label: str) -> html.Div:
    """
    Wrap an interactive graph with an accessible name for assistive technology.

    A labelled ``figure`` landmark, not ``role="img"``, which would hide the
    interactive canvas from assistive technology.

    Parameters
    ----------
    graph
        The graph to wrap (kept inside a ``dcc.Loading`` spinner).
    label
        The accessible name announced for the chart.

    Returns
    -------
    html.Div
        The loading-wrapped graph carrying an accessible name.
    """
    return html.Div(dcc.Loading(graph), role="figure", **{"aria-label": label})


def _panel_help(*paragraphs: str, summary: str = "How this works") -> html.Details:
    """
    Build a collapsed "how this works" disclosure for a panel's methodology prose.

    Each panel leads with its heading and chart; the long explanation lives here so the
    page reads as data first, prose on demand (styling via ``.mlpeg-explorer-help``).

    Parameters
    ----------
    *paragraphs
        One or more prose paragraphs to reveal when expanded.
    summary
        The always-visible toggle label.

    Returns
    -------
    html.Details
        The collapsed help block.
    """
    return html.Details(
        [html.Summary(summary), *[html.P(text) for text in paragraphs]],
        className="mlpeg-explorer-help",
    )


def _multi_select(
    component_id: str, values: list[str], placeholder: str
) -> dcc.Dropdown:
    """
    Build an empty multi-select over string values.

    Parameters
    ----------
    component_id
        The component id.
    values
        The selectable values (used as their own labels).
    placeholder
        Text shown while nothing is selected.

    Returns
    -------
    dcc.Dropdown
        The multi-select.
    """
    return dcc.Dropdown(
        id=component_id,
        options=[{"label": v, "value": v} for v in values],
        value=[],
        multi=True,
        placeholder=placeholder,
        # min() keeps a roomy width on desktop but shrinks to the container on
        # phones so a control never widens the page.
        style={"minWidth": "min(240px, 100%)", **_CONTROL_FONT},
    )


def _build_intro(n_rows: int, n_models: int, n_benchmarks: int) -> html.Div:
    """
    Build the Explorer intro: heading, stat chips and orientation copy.

    Parameters
    ----------
    n_rows
        Number of benchmark structures.
    n_models
        Number of evaluated models.
    n_benchmarks
        Number of scored benchmarks.

    Returns
    -------
    html.Div
        The intro section.
    """
    return html.Div(
        [
            html.H1("ML-PEG Model Explorer"),
            html.Div(
                [
                    _chip(f"{n_rows:,} observations with shared errors"),
                    _chip(f"{n_models} models with parity data"),
                    _chip(f"{n_benchmarks} scored benchmarks"),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            ),
            html.P(
                "Describe your system in the finder to get a ranked shortlist, or "
                "pick a model's report card to see where it works across chemistry — "
                "and whether each test is in-domain or an extrapolation. Then compare "
                "two models head-to-head, dig into the structures models fail on, or "
                "see which models fail alike.",
                style={
                    "maxWidth": "72ch",
                    "color": "var(--mlpeg-ink-2)",
                    "fontSize": "14px",
                },
            ),
        ]
    )


def _build_jump_nav() -> html.Nav:
    """
    Build the sticky section jump nav.

    A direct child of the page container, not of the intro: a sticky element can
    only travel within its parent, so inside the intro it would never pin.

    Returns
    -------
    html.Nav
        The jump nav.
    """
    return html.Nav(
        [
            html.A(label, href=f"#{anchor}", className="mlpeg-explorer-jump")
            for label, anchor in _EXPLORER_JUMP_NAV
        ],
        className="mlpeg-explorer-nav",
        **{"aria-label": "Explorer sections"},
    )


def _build_finder_panel(category_names: list[str], elements: list[str]) -> html.Div:
    """
    Build the model-finder panel (use-case controls, ranking chart, top picks).

    Parameters
    ----------
    category_names
        Benchmark categories for the fine-tune filter.
    elements
        Element symbols for the chemistry filter.

    Returns
    -------
    html.Div
        The finder panel.
    """
    finder_results = cached_finder_rankings(None, None, "any")
    struct_pct = round(100 * (1 - FINDER_BENCH_WEIGHT))
    return html.Div(
        [
            html.H3(
                "Model finder: which model for your experiment?",
            ),
            _panel_help(
                "Tell it what you're working on — the kind of experiment, the "
                "elements involved, and how much extrapolation you're willing to "
                "trust — and get a ranked shortlist with the evidence behind each "
                "pick. Click any bar or card to open the model's full report card "
                "below.",
            ),
            html.Div(
                _labelled_control(
                    "What are you simulating?",
                    _radio_items(FINDER_PRESET, FINDER_EXPERIMENTS, "any"),
                ),
                style=_ROW_GAP,
            ),
            html.Div(
                [
                    _labelled_control(
                        "Elements (match any selected)",
                        _multi_select(FINDER_ELEMENTS, elements, "Any chemistry"),
                    ),
                    _labelled_control(
                        "Benchmark areas (fine-tune)",
                        _multi_select(FINDER_CATEGORIES, category_names, "All areas"),
                    ),
                    _labelled_control(
                        "Evidence to trust",
                        _radio_items(FINDER_TOLERANCE, FINDER_TOLERANCE_OPTIONS, "any"),
                    ),
                ],
                className="mlpeg-explorer-controls",
            ),
            _a11y_graph(
                _graph(FINDER_GRAPH, build_finder_figure(finder_results)),
                "Model finder ranking bar chart. The labelled controls above are the "
                "keyboard route to the same ranking, and the pick cards below open a "
                "model's report card.",
            ),
            finder_legend(),
            html.Div(id=FINDER_PICKS, children=render_finder_picks(finder_results)),
            _panel_help(
                "Each model's fit combines two independent evidence streams. "
                "Benchmark evidence is its mean 0–1 score over the selected areas "
                "(only tests passing your extrapolation tolerance); each score is "
                "anchored to that benchmark's own good/bad error thresholds, so it "
                "is a relative grade, not an absolute error. When you pick elements, "
                "structure evidence uses observations containing any selected element. "
                "Models passing the tolerance are ranked by error, with tied errors "
                "sharing a rank. A model's mean win rate (averaged per "
                "benchmark so a large one can't dominate) "
                f"is blended {round(100 * FINDER_BENCH_WEIGHT)}% benchmark "
                f"+ {struct_pct}% structure into the fit — so it shifts with the set "
                f"of models compared. Models with fewer than {MIN_FINDER_BENCH} "
                "admissible benchmarks are too thin to rank; the rest have their mean "
                "gently pulled towards the field average in proportion to how few "
                "benchmarks back it, so a model can't win on a couple of lucky "
                "results (hover shows the benchmark count behind each). Physicality "
                "(does the model behave physically sensibly?) is always checked and "
                "flagged when poor — accuracy means little if a run blows up.",
                summary="How the ranking works",
            ),
        ],
        className=_PANEL_CLASS,
    )


def _build_report_panel() -> html.Div:
    """
    Build the model report-card panel (model picker, summary, profile, table).

    Returns
    -------
    html.Div
        The report-card panel, initialised to the broadest-coverage model.
    """
    default_model = report_default_model()
    return html.Div(
        [
            html.H3(
                "Model report card: where it works, and where it extrapolates",
            ),
            _panel_help(
                "Pick a model to see its score across every chemistry domain, with "
                "the all-model mean (grey tick) and the model's rank for context. "
                "Each category is coloured by whether the model is mostly tested "
                "in-domain (reference matches its training level of theory) or "
                "extrapolated — against a different functional, a higher-level "
                "reference (CCSD(T)/MP2), or experiment; hover for the exact "
                "breakdown. Scores are graded against each benchmark's own good/bad "
                "thresholds (not absolute errors), and a rank's cohort size varies by "
                "category, so treat close ranks as ties.",
            ),
            html.Div(
                _labelled_control(
                    "Model",
                    dcc.Dropdown(
                        id=REPORT_MODEL,
                        options=model_options(),
                        value=default_model,
                        clearable=False,
                        # Roomy on desktop (long model names), container-width on
                        # phones.
                        style={"minWidth": "min(420px, 100%)", **_CONTROL_FONT},
                    ),
                ),
                style=_ROW_GAP,
            ),
            html.Div(id=REPORT_SUMMARY, children=render_report_summary(default_model)),
            # Chart full-width, table full-width below: side-by-side the five-column
            # table overflowed its half of the card.
            _a11y_graph(
                _graph(REPORT_PROFILE, build_report_profile_figure(default_model)),
                "Report card profile: bar chart of the selected model's mean score in "
                "each chemistry area, against the all-model average. The table below "
                "lists every benchmark behind it.",
            ),
            lot_class_legend(),
            html.Details(
                [
                    html.Summary(
                        "Every benchmark for this model",
                        style={
                            "fontSize": "13px",
                            "fontWeight": "600",
                            "cursor": "pointer",
                            "margin": "8px 0 6px",
                        },
                    ),
                    html.Div(
                        id=REPORT_TABLE,
                        children=render_report_table(default_model),
                        className="mlpeg-report-table-wrap",
                    ),
                ],
                open=True,
            ),
        ],
        className=_PANEL_CLASS,
    )


def _build_h2h_panel(
    h2h_options: list[dict[str, str]],
    default_a: str | None,
    default_b: str | None,
) -> html.Div:
    """
    Build the head-to-head panel (model pickers, verdict bars, drilldown).

    Parameters
    ----------
    h2h_options
        Model dropdown options, labelled with each model's training level of theory.
    default_a, default_b
        The initially compared models.

    Returns
    -------
    html.Div
        The head-to-head panel.
    """
    stats = h2h_pair_stats(default_a, default_b)
    tie_pct = round(100 * H2H_TIE_BAND)
    return html.Div(
        [
            html.H3(
                "Head-to-head: model vs model",
            ),
            _panel_help(
                "One bar per benchmark area: it points towards the model that "
                "was closer to the reference more often, and its length is the margin. "
                "Each benchmark averages its metrics equally, and each area averages "
                "its benchmarks equally. Hover for comparison counts and exact ties. "
                f"Grey bars are within ±{tie_pct}% of even. Click a bar or pick an "
                "area to compare error magnitudes, then select a point to see its "
                "details and any matching 3D geometry.",
            ),
            html.Div(
                [
                    _labelled_control(
                        "Model A",
                        dcc.Dropdown(
                            id=H2H_MODEL_A,
                            options=h2h_options,
                            value=default_a,
                            clearable=False,
                            style={"minWidth": "min(260px, 100%)", **_CONTROL_FONT},
                        ),
                    ),
                    _labelled_control(
                        "Model B",
                        dcc.Dropdown(
                            id=H2H_MODEL_B,
                            options=h2h_options,
                            value=default_b,
                            clearable=False,
                            style={"minWidth": "min(260px, 100%)", **_CONTROL_FONT},
                        ),
                    ),
                ],
                className="mlpeg-explorer-controls",
            ),
            html.Div(
                id=H2H_SUMMARY,
                children=render_h2h_summary(default_a, default_b, stats),
                style=_ROW_GAP,
            ),
            _a11y_graph(
                _graph(
                    H2H_GRAPH, build_head_to_head_figure(default_a, default_b, stats)
                ),
                "Head-to-head verdict: diverging bar chart of which of the two chosen "
                "models is more accurate in each area, and by how much. Clicking a "
                "bar, or choosing an area in the picker below, opens that area's "
                "structures.",
            ),
            html.Div(
                _labelled_control(
                    "Open an area's structures",
                    dcc.Dropdown(
                        id=H2H_DETAIL_CATEGORY,
                        options=h2h_area_options(stats),
                        value=None,
                        clearable=True,
                        placeholder="Pick an area (or click a bar above)",
                        style={"minWidth": "min(300px, 100%)", **_CONTROL_FONT},
                    ),
                ),
                style={"marginTop": "var(--sp-3)", **_ROW_GAP},
            ),
            _a11y_graph(
                _graph(
                    H2H_DETAIL_GRAPH,
                    go.Figure(),
                    display="none",
                ),
                "Head-to-head detail: scatter of the two models' errors on every "
                "structure in the opened area. Clicking a point shows the structure "
                "in 3D; the structure list under the failure bars is the keyboard "
                "route to the same 3D panel.",
            ),
        ],
        className=_PANEL_CLASS,
    )


def _build_structure_fails_panel(
    category_names: list[str], focus_options: list[dict[str, str]]
) -> html.Div:
    """
    Build the structure-fails explorer (hardest structures + where a model fails).

    Parameters
    ----------
    category_names
        Benchmark categories for the area filter.
    focus_options
        Options for the "focus on one model" picker (family-sorted).

    Returns
    -------
    html.Div
        The structure-fails panel.
    """
    ranked = cached_structure_fails_ranking(None, None, "hardest")
    return html.Div(
        [
            html.H3(
                "Where models fail: the hardest structures",
            ),
            _panel_help(
                "The structures the field gets most wrong — ranked by how many "
                "models have an outlier error there (over "
                f"{OUTLIER_FACTOR:g}× their own median error on that benchmark, so "
                f"units don't matter; at most {FAILS_PER_BENCHMARK} per benchmark) "
                "or by how much the models disagree. Pick a single model to switch "
                "the ranking to where that model's error is the largest multiple of "
                "the median model's — the quickest way to see a model's weak spots. "
                "Click any bar "
                "(or pick from the list under the chart) to open its 3D geometry and "
                "every model's error in the panel below.",
            ),
            html.Div(
                [
                    _labelled_control(
                        "Benchmark areas",
                        _multi_select(
                            STRUCT_FAILS_CATEGORIES, category_names, "All areas"
                        ),
                    ),
                    _labelled_control(
                        "Focus on one model",
                        dcc.Dropdown(
                            id=STRUCT_FAILS_MODEL,
                            options=focus_options,
                            value=None,
                            clearable=True,
                            placeholder="All models (rank by difficulty)",
                            style={"minWidth": "min(360px, 100%)", **_CONTROL_FONT},
                        ),
                    ),
                    _labelled_control(
                        "Rank by",
                        _radio_items(
                            STRUCT_FAILS_SORT, STRUCT_FAILS_SORT_OPTIONS, "hardest"
                        ),
                    ),
                ],
                className="mlpeg-explorer-controls",
            ),
            _a11y_graph(
                _graph(STRUCT_FAILS_GRAPH, build_structure_fails_figure(ranked)),
                "Structure failures bar chart: the hardest or most-contested "
                "structures for the current selection. Clicking a bar, or a button "
                "in the list below, opens the structure in 3D with every model's "
                "error.",
            ),
            html.Div(id=STRUCT_FAILS_LIST, children=render_fails_list(ranked)),
        ],
        className=_PANEL_CLASS,
    )


def _build_detail_panel() -> html.Div:
    """
    Build the selected-structure panel (3D viewer + per-model error table slots).

    Returns
    -------
    html.Div
        The structure-detail panel, empty until a structure is opened.
    """
    return html.Div(
        [
            html.H3("Selected structure"),
            html.Div(
                "Open a structure from the head-to-head detail scatter or the "
                "failure bars above (click it, or use the list under the bars) to "
                "view its 3D geometry and every model's error here.",
                id=STRUCT_TABLE,
                style={"fontSize": "13px", "color": "var(--mlpeg-ink-2)"},
            ),
            dcc.Loading(html.Div(id=STRUCT_VIEWER)),
        ],
        className=_PANEL_CLASS,
    )


def _build_similarity_panel() -> html.Div:
    """
    Build the who-fails-alike panel (model x model error-correlation heatmap).

    Returns
    -------
    html.Div
        The similarity panel.
    """
    return html.Div(
        [
            html.H3(
                "Who fails alike: signed residual correlation",
            ),
            _panel_help(
                "Prediction minus reference is standardised per model and metric. "
                "Each cell correlates those signed residuals on shared observations. "
                "Hover for the shared count. Metrics with more shared observations "
                "contribute more points.",
            ),
            _a11y_graph(
                _graph(
                    SIMILARITY_GRAPH, build_similarity_heatmap_figure(load_similarity())
                ),
                "Model similarity heatmap: pairwise error correlation between models, "
                "clustered so models that fail alike group together.",
            ),
        ],
        className=_PANEL_CLASS,
    )


def _build_coverage_panel() -> html.Div:
    """
    Build the coverage-rose panel (field-wide progress overview).

    Returns
    -------
    html.Div
        The coverage panel.
    """
    return html.Div(
        [
            html.H3(
                "Coverage rose: mean scores for evaluated models",
            ),
            _panel_help(
                "Wedge length and colour show the mean score of the models evaluated "
                "on each benchmark. The area view averages its benchmarks equally. "
                "Model coverage and scoring thresholds vary by benchmark.",
            ),
            html.Div(
                _labelled_control(
                    "Group by",
                    dcc.Dropdown(
                        id=COVERAGE_GRANULARITY,
                        options=COVERAGE_GRANULARITY_OPTIONS,
                        value=COVERAGE_GRANULARITY_OPTIONS[0]["value"],
                        clearable=False,
                        style={"width": "min(260px, 100%)", **_CONTROL_FONT},
                    ),
                ),
                style=_ROW_GAP,
            ),
            _a11y_graph(
                _graph(
                    COVERAGE_GRAPH, build_coverage_rose_figure(load_benchmark_scores())
                ),
                "Coverage rose: mean scores for evaluated models by benchmark area.",
            ),
        ],
        className=_PANEL_CLASS,
    )


# (what it grounds, citation, url) for the methods panel
_REFERENCES: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Model finder",
        [
            (
                "structure evidence = mean win rate",
                "Liang et al. (2022), HELM",
                "https://arxiv.org/abs/2211.09110",
            ),
            (
                "aggregating scores into an MLIP leaderboard",
                "Riebesell et al. (2023), Matbench Discovery",
                "https://arxiv.org/abs/2308.14920",
            ),
        ],
    ),
    (
        "Head-to-head",
        [
            (
                "per-structure win/loss = the sign test",
                "Demšar (2006), JMLR 7:1",
                "https://www.jmlr.org/papers/v7/demsar06a.html",
            ),
        ],
    ),
    (
        "Model report card",
        [
            (
                "per-model reporting disaggregated by condition",
                "Mitchell et al. (2019), Model Cards",
                "https://arxiv.org/abs/1810.03993",
            ),
        ],
    ),
    (
        "Who fails alike",
        [
            (
                "error correlation as ensemble (dis)agreement",
                "Kuncheva & Whitaker (2003), Machine Learning 51:181",
                "https://doi.org/10.1023/A:1022859003006",
            ),
        ],
    ),
]


def _ref(grounds: str, citation: str, href: str) -> html.Li:
    """
    Render one reference line: what it grounds, then a linked citation.

    Parameters
    ----------
    grounds
        The part of a view this reference underpins.
    citation
        Short citation text (author, year, venue).
    href
        A URL that resolves to the work (arXiv/DOI/venue page).

    Returns
    -------
    html.Li
        The list item.
    """
    return html.Li(
        [
            html.Span(f"{grounds} — ", style={"color": "var(--mlpeg-ink-3)"}),
            html.A(
                citation,
                href=href,
                target="_blank",
                rel="noopener",
                style={"color": "var(--mlpeg-accent)"},
            ),
        ],
        style={"marginBottom": "4px"},
    )


def _build_references_panel() -> html.Div:
    """
    Build the collapsed "Methods & references" panel at the page bottom.

    Surfaces the literature each view is built on, so the analytics read as grounded
    rather than ad hoc. Content is driven by :data:`_REFERENCES`.

    Returns
    -------
    html.Div
        The references panel.
    """
    groups: list[html.Div] = []
    for view, refs in _REFERENCES:
        groups.append(
            html.Div(
                [
                    html.Div(
                        view,
                        style={
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "marginTop": "8px",
                        },
                    ),
                    html.Ul(
                        [_ref(*ref) for ref in refs],
                        style={
                            "margin": "2px 0 0",
                            "paddingLeft": "18px",
                            "fontSize": "12px",
                            "lineHeight": "1.5",
                        },
                    ),
                ]
            )
        )
    text = {
        "fontSize": "13px",
        "color": "var(--mlpeg-ink-2)",
        "maxWidth": "76ch",
    }
    return html.Div(
        html.Details(
            [
                html.Summary(
                    "How the scores work · methods & references",
                    style={
                        "fontSize": "16px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                    },
                ),
                html.P(
                    [
                        "Every number on this page traces back to one per-model, "
                        "per-benchmark ",
                        html.B("score in 0–1"),
                        ". Each benchmark measures one or more raw errors (e.g. mean "
                        "absolute error, in meV/atom, GPa or kcal/mol). Each raw error "
                        "is mapped to 0–1 by linear interpolation between a ",
                        html.B("good"),
                        " threshold (→ 1) and a ",
                        html.B("bad"),
                        " threshold (→ 0), clipped outside — the good/bad thresholds "
                        "are set per benchmark by domain experts (hover any metric for "
                        "its values). A benchmark's score is the weighted average of "
                        "its metric scores.",
                    ],
                    style={**text, "marginBottom": "6px"},
                ),
                html.P(
                    [
                        "So 1 means 'as accurate as anyone needs here' and 0 means 'at "
                        "or beyond the point where the model should be avoided'. "
                        "Everything else is built from these scores: the report card "
                        "and coverage rose average them, the finder blends a "
                        "benchmark-score mean with a win-rate over other models, and "
                        "the head-to-head counts per-structure wins. Full detail is in "
                        "the project's ",
                        html.A(
                            "scoring & normalisation guide",
                            href=SCORING_GUIDE_URL,
                            target="_blank",
                            rel="noopener",
                            style={"color": "var(--mlpeg-accent)"},
                        ),
                        ".",
                    ],
                    style={**text, "marginBottom": "6px"},
                ),
                html.P(
                    "Every view uses an established method — sources by view:",
                    style={**text, "marginBottom": "2px"},
                ),
                *groups,
            ],
            open=False,
        ),
        className=_PANEL_CLASS,
    )
