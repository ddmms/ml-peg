"""
Plotly figures and rendered fragments for the Explorer page.

Chrome colours are left unset for ``theme_plots.js`` to theme. Data colours that
mirror a theme token say so, as Plotly cannot read CSS variables server-side.
"""

from __future__ import annotations

import math
from typing import Any

from dash import dcc, html
import plotly.graph_objects as go

from ml_peg.analysis.explorer.build_explorer_data import OUTLIER_FACTOR
from ml_peg.app.utils.explorer_data import (
    category_page_paths,
    category_title,
    load_error_matrix,
    split_structure_key,
)
from ml_peg.app.utils.explorer_ids import (
    COVERAGE_GRANULARITY_OPTIONS,
    FAILS_PICK_TYPE,
    FINDER_PICK_TYPE,
    H2H_COLOUR_A,
    H2H_COLOUR_B,
    H2H_COLOUR_TIE,
    LOT_CLASS_STYLE,
)
from ml_peg.app.utils.explorer_scoring import (
    H2H_TIE_BAND,
    category_aggregates,
    category_model_means,
    h2h_verdict_counts,
    model_scores,
    ordinal,
    report_card_summary,
)
from ml_peg.app.utils.weas import generate_weas_html

# Figure heights (px). A graph's wrapper takes its height from the figure via
# graph_style(), so the two can never disagree.
FINDER_HEIGHT = 400
H2H_HEIGHT = 600
H2H_DETAIL_HEIGHT = 460
ROSE_HEIGHT = 560
_EMPTY_HEIGHT = 360
_REPORT_MIN_HEIGHT = 320
_FAILS_MIN_HEIGHT = 340
_HEATMAP_MIN_HEIGHT = 380
# Per-row heights for the figures that grow with their category/model count.
_BAR_ROW_PX = 26
_BAR_BASE_PX = 90
_HEATMAP_ROW_PX = 22
_HEATMAP_BASE_PX = 190
# Tallest a graph wrapper may be relative to the viewport, so a phone never shows a
# single chart taller than its screen.
_MAX_GRAPH_VH = 78

# How many entries the finder shows: bars in the chart, and evidence cards below it.
FINDER_TOP = 12
FINDER_PICKS_TOP = 3

# Name truncation for axis ticks / annotations and for structure labels.
_MODEL_NAME_LIMIT = 14
_SID_LABEL_LIMIT = 22

# Exact-zero errors cannot sit on a log axis; floor them at this fraction of the
# benchmark median.
_LOG_FLOOR_FRACTION = 0.01

# Mid-grey reference lines and tie band that read on both themes
_REFERENCE_GREY = "#8a8f98"
_REFERENCE_GREY_SOFT = "rgba(138,143,152,0.65)"
_TIE_BAND_FILL = LOT_CLASS_STYLE["none"]["color"]
_TIE_BAND_OPACITY = 0.08
# Where the "<model> more accurate →" annotations sit on the win-rate axis, and the
# axis half-width (a little past ±50% so the annotations clear the outer bars).
_H2H_ANNOTATION_X = 0.27
_H2H_AXIS_HALF_WIDTH = 0.56
# Gap between a bar (or its whisker / mean tick) and its value label, in axis units
_LABEL_PAD = 0.02
# Gap between a bar chart's category labels and the bars' baseline, in px
_TICK_STANDOFF_PX = 10

# The failure bars read red (bad), matching the app-wide bad/good score palette.
_FAILS_BAR_COLORSCALE = "Reds"

# Low (red) to high (green) mean score; stops mirror --mlpeg-bad and --mlpeg-warn
COVERAGE_COLORSCALE = [[0.0, "#e5484d"], [0.5, "#d97706"], [1.0, "#16a34a"]]

# Finder bars: normal pick (--mlpeg-accent), low physicality (--mlpeg-warn), and no
# structures containing the selected elements
_FINDER_COLOUR_NORMAL = LOT_CLASS_STYLE["in_domain"]["color"]
_FINDER_COLOUR_WARN = COVERAGE_COLORSCALE[1][1]
_FINDER_COLOUR_NO_EVIDENCE = LOT_CLASS_STYLE["none"]["color"]
_FINDER_LEGEND = [
    (_FINDER_COLOUR_NORMAL, "fit"),
    (_FINDER_COLOUR_WARN, "low physicality"),
    (_FINDER_COLOUR_NO_EVIDENCE, "no structures with your elements"),
]

# The 3D structure viewer, matching the benchmark pages' viewer (build_callbacks.py).
_VIEWER_HEIGHT_PX = 550


def graph_style(figure: go.Figure) -> dict[str, str]:
    """
    Size a ``dcc.Graph`` wrapper to its figure, capped to the viewport height.

    Parameters
    ----------
    figure
        The figure the graph will show.

    Returns
    -------
    dict[str, str]
        A ``style`` dict whose height matches ``figure.layout.height``.
    """
    height = int(figure.layout.height or _EMPTY_HEIGHT)
    return {"height": f"min({height}px, {_MAX_GRAPH_VH}vh)"}


def _short(name: str | None, limit: int = _MODEL_NAME_LIMIT) -> str:
    """
    Truncate a name for axis ticks and annotations.

    Parameters
    ----------
    name
        Model display name or structure id.
    limit
        Maximum length before an ellipsis.

    Returns
    -------
    str
        The (possibly truncated) name.
    """
    name = name or "?"
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _empty_figure(message: str, height: int = _EMPTY_HEIGHT) -> go.Figure:
    """
    Build a blank placeholder figure with a short centred message.

    Parameters
    ----------
    message
        The text to show.
    height
        Figure height in pixels.

    Returns
    -------
    go.Figure
        An axis-free figure with the message centred.
    """
    figure = go.Figure()
    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=message,
        font={"size": 13},
    )
    figure.update_layout(
        height=height,
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def _end_labels(
    x: list[float], y: list[Any], text: list[str], *, left: bool = False
) -> go.Scatter:
    """
    Build value labels that sit just past each bar, whisker or tick.

    A separate text trace, unlike bar ``textposition="outside"``, can be offset
    past whiskers and reference ticks. It skips hover, so it never sends clicks.

    Parameters
    ----------
    x
        Where each row's drawn content ends, before padding.
    y
        Row of each label.
    text
        Label text.
    left
        Place labels to the left of ``x`` (for bars that grow leftwards).

    Returns
    -------
    go.Scatter
        A text-only trace.
    """
    pad = -_LABEL_PAD if left else _LABEL_PAD
    return go.Scatter(
        x=[value + pad for value in x],
        y=y,
        text=text,
        mode="text",
        textposition="middle left" if left else "middle right",
        # No colour: inherit the themed layout font so labels read in dark mode.
        textfont={"size": 11},
        cliponaxis=False,
        hoverinfo="skip",
        showlegend=False,
    )


def _category_ticks(categories: list[str]) -> dict[str, Any]:
    """
    Show category titles on an axis whose data keeps directory names.

    Parameters
    ----------
    categories
        Category directory names on the axis.

    Returns
    -------
    dict[str, Any]
        ``tickvals`` and ``ticktext`` for a categorical axis.
    """
    return {
        "tickvals": categories,
        "ticktext": [category_title(c) for c in categories],
    }


def _count(n: int, noun: str) -> str:
    """
    Format a count with its (regular) noun, singular or plural.

    Parameters
    ----------
    n
        The count.
    noun
        The singular noun.

    Returns
    -------
    str
        E.g. ``"1 benchmark"`` or ``"3 benchmarks"``.
    """
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _glyph(symbol: str, meaning: str) -> html.Span:
    """
    Render a status glyph with a text alternative for assistive technology.

    Parameters
    ----------
    symbol
        The visible character (e.g. ``"⚠"``).
    meaning
        What the glyph conveys, announced instead of the symbol.

    Returns
    -------
    html.Span
        The glyph.
    """
    return html.Span(f"{symbol} ", role="img", **{"aria-label": meaning})


def _legend(entries: list[tuple[str, str]], caption: str) -> html.Div:
    """
    Render a compact colour key for a chart whose bars are colour-coded.

    Parameters
    ----------
    entries
        ``(colour, label)`` pairs in display order.
    caption
        Leading label naming what the colours encode.

    Returns
    -------
    html.Div
        The legend row.
    """
    return html.Div(
        [
            html.Span(caption, className="mlpeg-explorer-legend-caption"),
            *[
                html.Span(
                    [
                        html.Span(
                            "●",
                            style={"color": colour},
                            **{"aria-hidden": "true"},
                        ),
                        f" {label}",
                    ],
                    className="mlpeg-explorer-legend-item",
                )
                for colour, label in entries
            ],
        ],
        className="mlpeg-explorer-legend",
    )


def lot_class_legend() -> html.Div:
    """
    Render the level-of-theory colour key used by the report card.

    Returns
    -------
    html.Div
        The legend row.
    """
    return _legend(
        [(style["color"], style["label"]) for style in LOT_CLASS_STYLE.values()],
        "Test type:",
    )


def finder_legend() -> html.Div:
    """
    Render the colour key for the finder's ranking bars.

    Returns
    -------
    html.Div
        The legend row.
    """
    return _legend(_FINDER_LEGEND, "Bars:")


# --- Model report card ---------------------------------------------------------------


def build_report_profile_figure(model: str | None) -> go.Figure:
    """
    Build the report-card profile: the model's mean score per category, in context.

    Bars are coloured by level-of-theory class, with the all-model mean as a tick and
    the model's rank, so each score reads relative to the field.

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    go.Figure
        A horizontal bar figure (a placeholder if the model has no accuracy scores).
    """
    # Physicality is soundness, not accuracy -- it has its own summary chip, so keep it
    # off the accuracy profile to avoid conflating the two on one 0-1 axis.
    cats = [c for c in category_aggregates(model) if c["domain"] != "physicality"]
    if not cats:
        return _empty_figure(
            "No accuracy benchmarks for this model.", height=_REPORT_MIN_HEIGHT
        )
    cats.sort(key=lambda c: c["mean"])
    labels = [c["category"] for c in cats]
    values = [c["mean"] for c in cats]
    colours = [
        LOT_CLASS_STYLE.get(c["lot_class"], LOT_CLASS_STYLE["none"])["color"]
        for c in cats
    ]

    cohort = category_model_means()
    cohort_means: list[float | None] = []
    ranks: list[str] = []
    for c in cats:
        means = cohort.get(c["category"], {})
        cohort_means.append(sum(means.values()) / len(means) if means else None)
        if model in means:
            rank = 1 + sum(1 for v in means.values() if v > means[model])
            ranks.append(f"{ordinal(rank)} of {len(means)}")
        else:
            ranks.append("")

    customdata = [
        [c["lot_breakdown"], c["n"], rank, cohort_mean or 0, category_title(label)]
        for c, rank, cohort_mean, label in zip(
            cats, ranks, cohort_means, labels, strict=True
        )
    ]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colours},
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[4]}</b><br>mean score %{x:.2f} · "
                "rank %{customdata[2]}<br>"
                "all-model mean %{customdata[3]:.2f}<br>"
                "%{customdata[0]} · %{customdata[1]} benchmarks<extra></extra>"
            ),
        )
    )
    # All-model mean per category as a grey tick, so solved-vs-open areas are visible
    # on the same bars.
    tick_x = [m for m in cohort_means if m is not None]
    tick_y = [
        label for label, m in zip(labels, cohort_means, strict=True) if m is not None
    ]
    figure.add_trace(
        go.Scatter(
            x=tick_x,
            y=tick_y,
            mode="markers",
            marker={
                "symbol": "line-ns-open",
                "size": 14,
                "color": _REFERENCE_GREY,
                "line": {"width": 2},
            },
            name="all-model mean",
            hovertemplate="all-model mean %{x:.2f}<extra></extra>",
        )
    )
    # Rank labels clear both the bar and the mean tick, whichever ends further right
    label_x = [
        max(value, mean if mean is not None else value)
        for value, mean in zip(values, cohort_means, strict=True)
    ]
    figure.add_trace(_end_labels(label_x, labels, ranks))
    figure.update_layout(
        template="plotly_white",
        height=max(_REPORT_MIN_HEIGHT, _BAR_ROW_PX * len(cats) + _BAR_BASE_PX),
        margin={"l": 170, "r": 95, "t": 10, "b": 44},
        xaxis={"title": "mean score (0–1)", "range": [0, 1.08]},
        yaxis={
            "tickfont": {"size": 11},
            "automargin": True,
            "showgrid": False,
            "ticklabelstandoff": _TICK_STANDOFF_PX,
            "autorange": "reversed",  # weakest category first, at the top
            **_category_ticks(labels),
        },
        bargap=0.35,
        showlegend=False,
    )
    return figure


def _summary_chip(
    label: str, agg: tuple[float | None, int], lot_class: str, sub: str
) -> html.Div:
    """
    Render one report-card summary chip (headline number + context).

    Parameters
    ----------
    label
        Chip title.
    agg
        ``(mean, n)`` from :func:`report_card_summary`.
    lot_class
        LoT class whose accent colour the value takes.
    sub
        One-line explanation beneath the value.

    Returns
    -------
    html.Div
        The chip.
    """
    mean, n = agg
    colour = LOT_CLASS_STYLE.get(lot_class, LOT_CLASS_STYLE["none"])["color"]
    value = f"{mean:.2f}" if mean is not None else "—"
    return html.Div(
        [
            html.Div(label, className="mlpeg-report-chip-label"),
            html.Div(
                value, className="mlpeg-report-chip-value", style={"color": colour}
            ),
            html.Div(
                f"{_count(n, 'benchmark')} · {sub}", className="mlpeg-report-chip-sub"
            ),
        ],
        className="mlpeg-report-chip",
    )


def render_report_summary(model: str | None) -> html.Div:
    """
    Render the report-card extrapolation summary (training level + the three chips).

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    html.Div
        The summary chip row.
    """
    summary = report_card_summary(model)
    training = html.Div(
        [
            html.Div("Trained at", className="mlpeg-report-chip-label"),
            html.Div(
                summary["training_lot"] or "—",
                className="mlpeg-report-chip-value mlpeg-report-chip-value--lot",
            ),
            html.Div(
                f"{summary['n_benchmarks']} benchmarks evaluated",
                className="mlpeg-report-chip-sub",
            ),
        ],
        className="mlpeg-report-chip",
    )
    # Spell out what the extrapolated mean pools, so "0.63" isn't read as one kind of
    # test when it mixes functional-mismatch, CCSD(T)/MP2 and experimental references.
    kind_bits = " / ".join(
        f"{count} {LOT_CLASS_STYLE[cls]['label']}"
        for cls, count in sorted(
            summary["extrapolated_kinds"].items(), key=lambda kv: -kv[1]
        )
    )
    chips = html.Div(
        [
            training,
            _summary_chip(
                "In-domain accuracy",
                summary["in_domain"],
                "in_domain",
                "reference = training level",
            ),
            _summary_chip(
                "Extrapolated accuracy",
                summary["extrapolated"],
                "high_level",
                kind_bits or "reference ≠ training level",
            ),
            _summary_chip(
                "Physicality",
                summary["physicality"],
                "none",
                "sensible physical behaviour, independent of accuracy",
            ),
        ],
        className="mlpeg-report-summary",
    )
    # The methodology note sits under the chip row, not inside it, so it never reads
    # as a fifth (empty) chip.
    caption = html.Div(
        "Chip values are unweighted means of 0–1 benchmark scores.",
        className="mlpeg-report-chip-sub",
        style={"margin": "4px 0 8px"},
    )
    return html.Div([chips, caption])


def render_report_table(model: str | None) -> html.Table:
    """
    Render the per-benchmark table for the report card (best score first).

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    html.Table
        Benchmark · score · category · reference level · test-type (LoT class) rows.
    """
    rows = sorted(model_scores(model), key=lambda r: r["score"], reverse=True)
    # Score sits right after the benchmark name: long category names can push later
    # columns behind the horizontal scroll, and the score is the column that matters.
    header = html.Tr(
        [
            html.Th("Benchmark"),
            html.Th("Score"),
            html.Th("Category"),
            html.Th("Reference"),
            html.Th("Test"),
        ]
    )
    paths = category_page_paths()
    body = []
    for row in rows:
        style = LOT_CLASS_STYLE.get(row["lot_class"], LOT_CLASS_STYLE["none"])
        reference = ", ".join(row["ref_levels"]) if row["ref_levels"] else "—"
        path = paths.get(row["category"])
        title = category_title(row["category"])
        category_cell = dcc.Link(title, href=path) if path else title
        body.append(
            html.Tr(
                [
                    html.Td(row["benchmark"].split("/")[-1]),
                    html.Td(f"{row['score']:.2f}"),
                    html.Td(category_cell),
                    html.Td(reference, style={"fontSize": "11px"}),
                    html.Td(
                        [
                            html.Span(
                                "● ",
                                style={"color": style["color"]},
                                **{"aria-hidden": "true"},
                            ),
                            style["label"],
                        ]
                    ),
                ]
            )
        )
    return html.Table(
        [html.Thead(header), html.Tbody(body)],
        className="mlpeg-explorer-struct-table",
    )


# --- Head-to-head --------------------------------------------------------------------


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion ``k / n``.

    Structures within a benchmark are correlated, so this is a lower bound on the
    uncertainty.

    Parameters
    ----------
    k
        Number of successes (wins for model A).
    n
        Number of trials (shared structures in the category).
    z
        Normal quantile (1.96 = 95%).

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` bounds in ``[0, 1]``.
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def _h2h_message(stats: dict[str, Any] | None) -> str | None:
    """
    Explain why a head-to-head has nothing to draw, if so.

    Parameters
    ----------
    stats
        Head-to-head stats, or ``None`` when the selection is not a pair of models.

    Returns
    -------
    str | None
        The placeholder message, or ``None`` when there is a comparison to show.
    """
    if stats is None:
        return "Pick two different models to compare."
    if not stats["n_shared"] or not stats["per_category"]:
        return "These two models share no structures."
    return None


def build_head_to_head_figure(
    model_a: str | None, model_b: str | None, stats: dict[str, Any] | None
) -> go.Figure:
    """
    Build the head-to-head verdict: diverging per-category win-rate bars.

    One bar per category, right (blue) when ``model_a`` wins more structures and left
    (amber) when ``model_b`` does, with a grey band for practical ties.

    Parameters
    ----------
    model_a, model_b
        Model names to compare (A = right of centre, B = left).
    stats
        Head-to-head stats for the pair (:func:`h2h_pair_stats`), or ``None`` when
        the selection is not two distinct models.

    Returns
    -------
    go.Figure
        A horizontal bar figure keyed by category on the y axis (so ``clickData``
        identifies the category to drill into), or a placeholder with the reason.
    """
    message = _h2h_message(stats)
    if message:
        return _empty_figure(message, height=H2H_HEIGHT)
    ordered = sorted(
        stats["per_category"], key=lambda row: row["win_rate_a"], reverse=True
    )

    short_a, short_b = _short(model_a), _short(model_b)
    margins = [row["win_rate_a"] - 0.5 for row in ordered]
    # Wilson half-widths from the decisive comparisons, drawn around the bar as a
    # sample-size cue
    err_plus = []
    err_minus = []
    for row in ordered:
        decisive = row["decisive"]
        if decisive:
            lo, hi = _wilson_interval(row["wins_a"], decisive)
            pooled = row["wins_a"] / decisive
            err_plus.append(hi - pooled)
            err_minus.append(pooled - lo)
        else:
            err_plus.append(0.0)
            err_minus.append(0.0)
    colours = []
    for margin in margins:
        if margin > H2H_TIE_BAND:
            colours.append(H2H_COLOUR_A)
        elif margin < -H2H_TIE_BAND:
            colours.append(H2H_COLOUR_B)
        else:
            colours.append(H2H_COLOUR_TIE)

    categories = [row["category"] for row in ordered]
    figure = go.Figure(
        go.Bar(
            x=margins,
            y=categories,
            orientation="h",
            marker={"color": colours},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": err_plus,
                "arrayminus": err_minus,
                "thickness": 1,
                "width": 3,
                "color": _REFERENCE_GREY_SOFT,
            },
            customdata=[
                [
                    100 * row["win_rate_a"],
                    row["n"],
                    row["n_benchmarks"],
                    category_title(row["category"]),
                ]
                for row in ordered
            ],
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                f"{short_a} more accurate in %{{customdata[0]:.0f}}% "
                "(evenly weighted across %{customdata[2]} benchmark(s),<br>"
                "%{customdata[1]:,} structures)<br>"
                "click to see every structure<extra></extra>"
            ),
        )
    )
    # Practical-tie band and centre line, so near-even bars aren't over-read.
    figure.add_shape(
        type="rect",
        x0=-H2H_TIE_BAND,
        x1=H2H_TIE_BAND,
        xref="x",
        y0=0,
        y1=1,
        yref="paper",
        fillcolor=_TIE_BAND_FILL,
        opacity=_TIE_BAND_OPACITY,
        line={"width": 0},
        layer="below",
    )
    figure.add_vline(x=0, line={"color": _REFERENCE_GREY, "width": 1})
    # Labels sit past the whisker end, on the side the bar grows towards
    texts = [f"{100 * row['win_rate_a']:.0f}% · n={row['n']:,}" for row in ordered]
    for left in (False, True):
        rows = [i for i, margin in enumerate(margins) if (margin < 0) == left]
        if not rows:
            continue
        ends = [
            margins[i] - err_minus[i] if left else margins[i] + err_plus[i]
            for i in rows
        ]
        figure.add_trace(
            _end_labels(
                ends,
                [categories[i] for i in rows],
                [texts[i] for i in rows],
                left=left,
            )
        )
    figure.add_annotation(
        x=_H2H_ANNOTATION_X,
        y=1.05,
        xref="x",
        yref="paper",
        showarrow=False,
        text=f"<b>{short_a} more accurate →</b>",
        font={"size": 11, "color": H2H_COLOUR_A},
    )
    figure.add_annotation(
        x=-_H2H_ANNOTATION_X,
        y=1.05,
        xref="x",
        yref="paper",
        showarrow=False,
        text=f"<b>← {short_b} more accurate</b>",
        font={"size": 11, "color": H2H_COLOUR_B},
    )

    figure.update_layout(
        margin={"l": 170, "r": 120, "t": 42, "b": 55},
        height=H2H_HEIGHT,
        template="plotly_white",
        showlegend=False,
        xaxis={
            "title": (f"{short_a} win rate (benchmark-averaged, per area)"),
            "range": [-_H2H_AXIS_HALF_WIDTH, _H2H_AXIS_HALF_WIDTH],
            "tickvals": [-0.25, 0, 0.25],
            "ticktext": ["25%", "50% · even", "75%"],
            "tickfont": {"size": 10},
        },
        yaxis={
            "tickfont": {"size": 11},
            "autorange": "reversed",  # biggest A-wins on top (tornado order)
            "automargin": True,  # widen the left margin to fit long model names
            "showgrid": False,
            "ticklabelstandoff": _TICK_STANDOFF_PX,
            **_category_ticks(categories),
        },
        bargap=0.3,
    )
    return figure


def render_h2h_summary(
    model_a: str | None, model_b: str | None, stats: dict[str, Any] | None
) -> Any:
    """
    Render the head-to-head headline: who leads where, in plain words.

    Parameters
    ----------
    model_a, model_b
        The compared models.
    stats
        Head-to-head stats for the pair (:func:`h2h_pair_stats`), or ``None`` when
        the selection is not two distinct models.

    Returns
    -------
    Any
        The headline children (a sentence when there is nothing to compare).
    """
    message = _h2h_message(stats)
    if message:
        return message
    cats_a, cats_b, cats_tie = h2h_verdict_counts(stats)
    return html.Div(
        [
            html.Span(
                f"{model_a} ",
                style={"fontWeight": "600", "color": H2H_COLOUR_A},
            ),
            html.Span(f"leads in {cats_a} areas  ·  "),
            html.Span(
                f"{model_b} ",
                style={"fontWeight": "600", "color": H2H_COLOUR_B},
            ),
            html.Span(
                f"in {cats_b}  ·  {cats_tie} too close to call  —  "
                f"averaged evenly across areas, {model_a} was more "
                f"accurate {100 * stats['overall_win_rate_a']:.0f}% of the time "
                f"(over {stats['n_shared']:,} shared structures)"
            ),
        ],
        style={"fontSize": "13px"},
    )


def build_h2h_detail_figure(
    model_a: str,
    model_b: str,
    category: str,
    error_matrix: dict[str, dict[str, float]],
    meta: dict[str, dict[str, Any]],
    medians: dict[str, float],
) -> go.Figure:
    """
    Build one category's structure-level A-vs-B error scatter (the drilldown).

    One point per shared structure against a ``y = x`` diagonal. Errors are divided by
    their benchmark's median error, so benchmarks with different units share axes.

    Parameters
    ----------
    model_a, model_b
        Model names on the x and y axes.
    category
        Benchmark category to restrict to.
    error_matrix
        ``structure_key -> {model: signed_residual}`` from :func:`load_error_matrix`.
    meta
        ``structure_key -> hardness row`` (:func:`load_hardness_by_key`), used to
        attach benchmark/sid to each point for hover.
    medians
        ``benchmark -> median |error|`` (:func:`benchmark_error_medians`).

    Returns
    -------
    go.Figure
        A ``scattergl`` figure whose points carry ``[key, sid, benchmark]`` in
        ``customdata``; the key is what the shared 3D drilldown needs.
    """
    xs: list[float] = []
    ys: list[float] = []
    colours: list[str] = []
    customdata: list[list[Any]] = []
    wins_a = wins_b = ties = 0
    for key, row in meta.items():
        if row["category"] != category:
            continue
        resids = error_matrix.get(key)
        if not resids or model_a not in resids or model_b not in resids:
            continue
        med = medians.get(split_structure_key(key)[1]) or 1.0
        floor = _LOG_FLOOR_FRACTION * med
        ea, eb = abs(resids[model_a]), abs(resids[model_b])
        xs.append(max(ea, floor) / med)
        ys.append(max(eb, floor) / med)
        if ea < eb:
            colours.append(H2H_COLOUR_A)
            wins_a += 1
        elif eb < ea:
            colours.append(H2H_COLOUR_B)
            wins_b += 1
        else:
            colours.append(H2H_COLOUR_TIE)
            ties += 1
        customdata.append([key, row["sid"], row["benchmark"]])
    if not xs:
        return _empty_figure(
            "These two models share no structures here.", height=H2H_DETAIL_HEIGHT
        )

    short_a, short_b = _short(model_a), _short(model_b)
    lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="markers",
            marker={"size": 6, "color": colours, "opacity": 0.7, "line": {"width": 0}},
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[1]}</b> (%{customdata[2]})<br>"
                f"{short_a}: %{{x:.2f}}× benchmark median<br>"
                f"{short_b}: %{{y:.2f}}× benchmark median<br>"
                "click for 3D structure<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": _REFERENCE_GREY, "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Winner halves in plain words: above the diagonal, B's error is larger.
    figure.add_annotation(
        x=0.03,
        y=0.97,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        text=f"<b>{short_a} more accurate</b>",
        font={"size": 11, "color": H2H_COLOUR_A},
    )
    figure.add_annotation(
        x=0.97,
        y=0.03,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="right",
        text=f"<b>{short_b} more accurate</b>",
        font={"size": 11, "color": H2H_COLOUR_B},
    )
    figure.update_layout(
        title={
            "text": (
                f"{category} — {short_a} wins {wins_a:,} · "
                f"{short_b} wins {wins_b:,} · {ties:,} ties"
            ),
            "font": {"size": 13},
        },
        margin={"l": 70, "r": 20, "t": 45, "b": 55},
        height=H2H_DETAIL_HEIGHT,
        template="plotly_white",
        showlegend=False,
        xaxis={"title": f"{short_a} error (× benchmark median)", "type": "log"},
        yaxis={"title": f"{short_b} error (× benchmark median)", "type": "log"},
    )
    return figure


# --- Model finder --------------------------------------------------------------------


def build_finder_figure(
    results: list[dict[str, Any]], top: int = FINDER_TOP
) -> go.Figure:
    """
    Render the model-finder ranking as a horizontal bar chart with evidence in hover.

    Parameters
    ----------
    results
        Ranked rows from :func:`model_finder_rankings`.
    top
        How many models to show.

    Returns
    -------
    go.Figure
        Bars best-first (top of chart); clicking a bar loads that model's report card.
    """
    shown = results[:top]
    if not shown:
        return _empty_figure(
            "No model has enough evidence for this selection — "
            "widen the categories or tolerance.",
            height=FINDER_HEIGHT,
        )

    shown = list(reversed(shown))  # best at the top of a horizontal bar chart
    hover: list[str] = []
    colours: list[str] = []
    for row in shown:
        bits = [
            f"benchmark score {row['bench_mean']:.2f} ({row['n_bench']} benchmarks)"
        ]
        if row["struct_score"] is not None:
            bits.append(
                f"element evidence {row['struct_score']:.2f} "
                f"({row['n_struct']:,} structures)"
            )
        if row["no_element_evidence"]:
            bits.append("no tested structures contain your elements")
        if row.get("physicality") is not None:
            bits.append(f"physicality {row['physicality']:.2f}")
        if row.get("low_physicality"):
            bits.append("low physicality — may behave unphysically")
        hover.append("<br>".join(bits))
        if row["no_element_evidence"]:
            colours.append(_FINDER_COLOUR_NO_EVIDENCE)
        elif row.get("low_physicality"):
            colours.append(_FINDER_COLOUR_WARN)
        else:
            colours.append(_FINDER_COLOUR_NORMAL)

    figure = go.Figure(
        go.Bar(
            x=[row["fit"] for row in shown],
            y=[row["model"] for row in shown],
            orientation="h",
            marker={"color": colours},
            customdata=hover,
            hovertemplate="<b>%{y}</b><br>fit %{x:.2f}<br>%{customdata}<extra></extra>",
        )
    )
    figure.add_trace(
        _end_labels(
            [row["fit"] for row in shown],
            [row["model"] for row in shown],
            [f"{row['fit']:.2f}" for row in shown],
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=FINDER_HEIGHT,
        margin={"l": 190, "r": 50, "t": 10, "b": 44},
        showlegend=False,
        xaxis={"title": "fit for your selection (0–1)", "range": [0, 1.08]},
        yaxis={
            "tickfont": {"size": 11},
            "automargin": True,
            "showgrid": False,
            "ticklabelstandoff": _TICK_STANDOFF_PX,
        },
        bargap=0.3,
    )
    return figure


def render_finder_picks(
    results: list[dict[str, Any]], top: int = FINDER_PICKS_TOP
) -> html.Div:
    """
    Render the finder's top picks as evidence cards ("why this model?").

    Cards are buttons that open the model's report card.

    Parameters
    ----------
    results
        Ranked rows from :func:`model_finder_rankings`.
    top
        How many picks to show.

    Returns
    -------
    html.Div
        The card row (empty Div when there are no results).
    """
    if not results:
        return html.Div()
    cards = []
    for i, row in enumerate(results[:top]):
        lines: list[Any] = [
            html.Div(
                [
                    html.Span(f"#{i + 1}  ", style={"color": "var(--mlpeg-ink-3)"}),
                    html.Span(row["model"], style={"fontWeight": "700"}),
                    html.Span(
                        f"  ·  fit {row['fit']:.2f}",
                        style={"color": "var(--mlpeg-accent)", "fontWeight": "600"},
                    ),
                ],
                # Break long model ids inside the card so they don't force its width.
                style={
                    "fontSize": "14px",
                    "marginBottom": "6px",
                    "overflowWrap": "anywhere",
                },
            ),
            html.Div(
                f"Trained at {row['training_lot'] or 'unknown level'}",
                className="mlpeg-report-chip-sub",
            ),
            html.Div(
                f"Benchmark evidence: {row['bench_mean']:.2f} over "
                f"{row['n_bench']} benchmarks "
                f"({100 * row['in_domain_frac']:.0f}% in-domain)",
                className="mlpeg-report-chip-sub",
            ),
        ]
        if row["struct_score"] is not None:
            lines.append(
                html.Div(
                    f"Your elements: typically more accurate than "
                    f"{100 * row['struct_score']:.0f}% of models across "
                    f"{row['n_struct']:,} test structures",
                    className="mlpeg-report-chip-sub",
                )
            )
        elif row["no_element_evidence"]:
            lines.append(
                html.Div(
                    [
                        _glyph("⚠", "warning"),
                        "No tested structures contain your elements",
                    ],
                    className="mlpeg-report-chip-sub",
                    style={"color": "var(--mlpeg-warn)"},
                )
            )
        if row["physicality"] is not None:
            if row["low_physicality"]:
                phys_note: list[Any] = [
                    " ",
                    _glyph("⚠", "warning"),
                    "may behave unphysically",
                ]
                phys_style = {"color": "var(--mlpeg-warn)"}
            else:
                phys_note = [" ", _glyph("✓", "sound")]
                phys_style = {}
            lines.append(
                html.Div(
                    [f"Physicality: {row['physicality']:.2f}", *phys_note],
                    className="mlpeg-report-chip-sub",
                    style=phys_style,
                )
            )
        lines.append(
            html.Div(
                "Open report card →",
                style={
                    "fontSize": "11px",
                    "color": "var(--mlpeg-accent)",
                    "marginTop": "6px",
                },
            )
        )
        cards.append(
            html.Button(
                lines,
                id={"type": FINDER_PICK_TYPE, "model": row["model"]},
                type="button",
                className="mlpeg-report-chip mlpeg-explorer-pick",
                n_clicks=0,
            )
        )
    return html.Div(cards, className="mlpeg-explorer-picks")


# --- Coverage rose -------------------------------------------------------------------


def _coverage_wedges(
    rows: list[dict[str, Any]], granularity: str
) -> list[dict[str, Any]]:
    """
    Reduce benchmark rows to the wedges the rose draws, ordered for a clean ring.

    Parameters
    ----------
    rows
        Benchmark rows from :func:`load_benchmark_scores`.
    granularity
        ``"category"`` collapses each category to one wedge (mean over its benchmarks);
        ``"benchmark"`` keeps one wedge per benchmark, ordered by category so a
        category's benchmarks sit contiguously on the ring.

    Returns
    -------
    list[dict[str, Any]]
        Wedges with ``label``, ``category``, ``mean_score`` and ``n``, ordered.
    """
    if granularity == "category":
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["category"], []).append(row)
        wedges = [
            {
                "label": category_title(category),
                "category": category,
                "mean_score": sum(r["mean_score"] for r in items) / len(items),
                "n": len(items),
            }
            for category, items in grouped.items()
        ]
        return sorted(wedges, key=lambda w: w["label"])

    return [
        {
            "label": row["benchmark"].split("/")[-1],
            "category": row["category"],
            "mean_score": row["mean_score"],
            "n": row["n_models"],
        }
        for row in sorted(rows, key=lambda r: (r["category"], r["benchmark"]))
    ]


def build_coverage_rose_figure(
    rows: list[dict[str, Any]],
    granularity: str = COVERAGE_GRANULARITY_OPTIONS[0]["value"],
) -> go.Figure:
    """
    Build the coverage rose: a polar bar chart of mean model score per area.

    Each wedge shows the mean score over all models for one area.

    Parameters
    ----------
    rows
        Benchmark rows from :func:`load_benchmark_scores`.
    granularity
        ``"category"`` (one wedge per category) or ``"benchmark"`` (one per benchmark).

    Returns
    -------
    go.Figure
        A ``Barpolar`` figure, or a placeholder if no scores are available.
    """
    wedges = _coverage_wedges(rows, granularity)
    if not wedges:
        return _empty_figure("No benchmark scores are available.", height=ROSE_HEIGHT)

    labels = [w["label"] for w in wedges]
    scores = [w["mean_score"] for w in wedges]
    n_unit = "models" if granularity == "benchmark" else "benchmarks"
    customdata = [[category_title(w["category"]), w["n"]] for w in wedges]
    # A category wedge is already labelled with its title
    area = "" if granularity == "category" else " (%{customdata[0]})"

    figure = go.Figure(
        go.Barpolar(
            r=scores,
            theta=labels,
            marker={
                "color": scores,
                "colorscale": COVERAGE_COLORSCALE,
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {
                    "title": {"text": "mean score, all models", "side": "right"},
                    "tickvals": [0, 0.5, 1],
                    "ticktext": ["0 · open problem", "0.5", "1 · solved"],
                },
                # theme_plots.js does not repaint marker lines, so use mid-grey
                "line": {"color": _REFERENCE_GREY_SOFT, "width": 1},
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{theta}</b>" + area + "<br>"
                "mean score across models: %{r:.2f}<br>"
                "%{customdata[1]} " + n_unit + "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=ROSE_HEIGHT,
        margin={"l": 40, "r": 40, "t": 20, "b": 20},
        showlegend=False,
        polar={
            "radialaxis": {
                "range": [0, 1],
                "tickvals": [0.25, 0.5, 0.75, 1.0],
                "angle": 90,
                # Tick labels would sit on a wedge; the colourbar carries the scale
                "showticklabels": False,
            },
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickfont": {"size": 11},
            },
        },
    )
    return figure


# --- Where models fail + selected structure ------------------------------------------


def build_structure_fails_figure(
    ranked: list[dict[str, Any]], model: str | None = None, sort: str = "hardest"
) -> go.Figure:
    """
    Build the structure-fails bars: the hardest structures, worst at the top.

    Clicking a bar opens the structure in the detail panel.

    Parameters
    ----------
    ranked
        Worst-first rows from :func:`structure_fails_ranking`.
    model
        The focused model, if any (names the axis).
    sort
        ``"hardest"`` or ``"contested"`` (names the axis when no model is focused).

    Returns
    -------
    go.Figure
        A horizontal bar figure whose bars carry ``[key, sid, benchmark, ...]`` in
        ``customdata`` (the key feeds the shared 3D drilldown, like the head-to-head
        detail scatter).
    """
    if not ranked:
        return _empty_figure(
            "No structures match this selection.", height=_FAILS_MIN_HEIGHT
        )
    if model:
        axis_title = f"{_short(model)} error ÷ median model error on the structure"
        score_line = f"{_short(model)}: %{{customdata[8]:.3g}}× the median model<br>"
    elif sort == "contested":
        axis_title = "model disagreement (coefficient of variation)"
        score_line = ""
    else:
        axis_title = "share of models with an outlier error here (0–1)"
        score_line = (
            "%{customdata[8]:.0%} of models err over "
            f"{OUTLIER_FACTOR:g}× their usual on this benchmark<br>"
        )
    ratio_axis = bool(model)
    xvals = [row["fail_score"] for row in ranked]
    # A leading rank number keeps y labels unique even when two benchmarks share a sid.
    labels = [
        f"{i + 1}. {_short(row['sid'], _SID_LABEL_LIMIT)} · "
        f"{_short(row['benchmark'], _SID_LABEL_LIMIT)}"
        for i, row in enumerate(ranked)
    ]
    customdata = [
        [
            row["key"],
            row["sid"],
            row["benchmark"],
            row["category"],
            row["median_err"],
            row["disagreement"],
            row["worst_model"],
            row["n_models"],
            row["fail_score"],
        ]
        for row in ranked
    ]
    figure = go.Figure(
        go.Bar(
            x=xvals,
            y=labels,
            orientation="h",
            marker={
                "color": xvals,
                "colorscale": _FAILS_BAR_COLORSCALE,
                "cmin": 0,
                "cmax": max(xvals) or 1.0,
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "%{customdata[3]} · %{customdata[2]}<br>"
                + score_line
                + "median |error| across %{customdata[7]} models: "
                "%{customdata[4]:.4g}<br>"
                "disagreement (CoV): %{customdata[5]:.2f}<br>"
                "worst model here: %{customdata[6]}<br>"
                "click to inspect in 3D<extra></extra>"
            ),
        )
    )
    if not (ratio_axis or sort == "contested"):
        # Spell out the share as a count, so ties at 1.0 read as "every model"
        figure.add_trace(
            _end_labels(
                xvals,
                labels,
                [f"{row['n_failing']}/{row['n_judged']} models" for row in ranked],
            )
        )
    figure.update_layout(
        margin={"l": 230, "r": 80, "t": 20, "b": 46},
        height=max(_FAILS_MIN_HEIGHT, _BAR_ROW_PX * len(ranked) + _BAR_BASE_PX),
        template="plotly_white",
        showlegend=False,
        xaxis={
            "title": axis_title,
            "tickfont": {"size": 10},
            "rangemode": "tozero",
            # Room for the "n/N models" labels past a full-width bar
            **(
                {}
                if ratio_axis or sort == "contested"
                else {"range": [0, 1.2], "tickvals": [0, 0.2, 0.4, 0.6, 0.8, 1]}
            ),
        },
        yaxis={
            "tickfont": {"size": 10},
            "automargin": True,
            "showgrid": False,
            "ticklabelstandoff": _TICK_STANDOFF_PX,
            "autorange": "reversed",  # worst (rank 1) at the top
        },
        bargap=0.28,
    )
    if ratio_axis:
        # A bar past 1x is worse than the median model on that structure
        figure.add_vline(
            x=1,
            line={"dash": "dash", "width": 1, "color": _REFERENCE_GREY_SOFT},
            annotation_text="1×",
            annotation_position="top",
            annotation_font_size=10,
        )
    return figure


def render_fails_list(ranked: list[dict[str, Any]]) -> html.Div:
    """
    Render the ranked structures as a list of buttons, for keyboard users.

    Keyboard-accessible buttons for the same structures as the bars.

    Parameters
    ----------
    ranked
        Worst-first rows from :func:`structure_fails_ranking`.

    Returns
    -------
    html.Div
        The disclosure (empty when there is nothing ranked).
    """
    if not ranked:
        return html.Div()
    items = [
        html.Li(
            html.Button(
                f"{row['sid']} · {row['category']} / {row['benchmark']}",
                id={"type": FAILS_PICK_TYPE, "key": row["key"]},
                type="button",
                className="mlpeg-explorer-list-btn",
                n_clicks=0,
            )
        )
        for row in ranked
    ]
    return html.Div(
        html.Details(
            [
                html.Summary("Open a structure from a list"),
                html.Ol(items, className="mlpeg-explorer-list"),
            ],
            className="mlpeg-explorer-help",
        )
    )


def _struct_table(key: str) -> html.Table:
    """
    Build a per-model error table for one structure, best model first.

    Parameters
    ----------
    key
        Structure key into the error matrix.

    Returns
    -------
    html.Table
        Table of model and absolute error (with signed residual), sorted ascending.
    """
    resids = load_error_matrix().get(key, {})
    ordered = sorted(resids.items(), key=lambda kv: abs(kv[1]))
    header = html.Tr([html.Th("Model"), html.Th("Abs. error"), html.Th("Signed")])
    body = [
        html.Tr(
            [
                html.Td(model),
                html.Td(f"{abs(resid):.4g}"),
                html.Td(f"{resid:+.4g}"),
            ]
        )
        for model, resid in ordered
    ]
    return html.Table(
        [html.Thead(header), html.Tbody(body)],
        className="mlpeg-explorer-struct-table",
    )


def render_structure_detail(key: str, row: dict[str, Any]) -> tuple[Any, Any]:
    """
    Render one structure's 3D geometry (WEAS) and its per-model error table.

    Shared by every route into the detail panel: the head-to-head scatter, the failure
    bars and their keyboard list.

    Parameters
    ----------
    key
        Structure key into the error matrix.
    row
        The structure's hardness row (benchmark, sid, geometry path).

    Returns
    -------
    tuple[Any, Any]
        The viewer children and the detail-table children.
    """
    category, _, _ = split_structure_key(key)
    path = category_page_paths().get(category)
    heading: list[Any] = [f"{row['sid']}  ·  {row['benchmark']}"]
    if path:
        heading.extend(
            [
                "  ·  ",
                dcc.Link("full benchmark page", href=path, style={"fontSize": "12px"}),
            ]
        )
    table = html.Div(
        [
            html.Div(heading, style={"fontWeight": "600", "marginBottom": "6px"}),
            _struct_table(key),
        ]
    )
    xyz = row.get("xyz")
    if not xyz:
        viewer = html.Div(
            "No 3D geometry is shipped for this structure.",
            style={
                "fontSize": "13px",
                "color": "var(--mlpeg-ink-3)",
                "marginTop": "8px",
            },
        )
        return viewer, table
    # Same viewer sizing and border as the benchmark pages' structure viewer, so the
    # two read as one component; geometries are served from the assets folder.
    viewer = html.Iframe(
        title=f"Interactive 3D structure viewer for {row['sid']}",
        srcDoc=generate_weas_html(f"/assets/{xyz}", "struct", 0),
        style={
            "height": f"{_VIEWER_HEIGHT_PX}px",
            "width": "100%",
            "border": "1px solid var(--mlpeg-border)",
            "borderRadius": "5px",
            "marginTop": "8px",
        },
    )
    return viewer, table


# --- Who fails alike -----------------------------------------------------------------


def build_similarity_heatmap_figure(sim: dict[str, Any]) -> go.Figure:
    """
    Build the who-fails-alike heatmap: model x model error correlation, clustered.

    Rows and columns follow the hierarchical clustering order.

    Parameters
    ----------
    sim
        The similarity payload from :func:`load_similarity`.

    Returns
    -------
    go.Figure
        The heatmap, or a short placeholder when too few models share evaluations.
    """
    models = sim.get("models", [])
    corr = sim.get("corr", [])
    if len(models) < 2 or not corr:
        return _empty_figure(
            "Not enough shared evaluations to compare models yet.",
            height=_HEATMAP_MIN_HEIGHT,
        )
    order = sim.get("order") or list(range(len(models)))
    labels = [models[k] for k in order]  # full names stay unique (heatmap axes need it)
    z = [[corr[oi][oj] for oj in order] for oi in order]
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu",
            reversescale=True,  # high correlation -> red (these two fail alike)
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            hovertemplate=(
                "%{y}<br>vs %{x}<br>error correlation: %{z:.2f}<extra></extra>"
            ),
            colorbar={"title": "corr", "thickness": 12, "tickfont": {"size": 9}},
        )
    )
    figure.update_layout(
        margin={"l": 150, "r": 20, "t": 20, "b": 150},
        height=max(
            _HEATMAP_MIN_HEIGHT, _HEATMAP_ROW_PX * len(labels) + _HEATMAP_BASE_PX
        ),
        template="plotly_white",
        xaxis={"tickangle": -45, "tickfont": {"size": 9}, "showgrid": False},
        yaxis={"tickfont": {"size": 9}, "autorange": "reversed", "showgrid": False},
    )
    return figure
