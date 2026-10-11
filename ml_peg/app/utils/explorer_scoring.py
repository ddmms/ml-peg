"""
Scoring and statistics for the Explorer page.

Pure functions with no Dash or Plotly imports, so they are testable without a browser.
The page's "Methods & references" panel cites the source of each method.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import statistics
from typing import Any

from ml_peg.app.utils.explorer_data import (
    load_error_matrix,
    load_hardness,
    load_model_benchmark_scores,
    model_coverage,
    model_taxonomy,
    split_structure_key,
    structure_model_ranks,
)
from ml_peg.app.utils.explorer_ids import LOT_CLASS_STYLE

# Win rates within this band of 50% are shown as a practical tie
H2H_TIE_BAND = 0.05

# Model-finder extrapolation tolerance -> admissible LoT classes (values of
# FINDER_TOLERANCE_OPTIONS in explorer_ids).
_TOLERANCE_CLASSES = {
    "in_domain": {"in_domain"},
    "dft": {"in_domain", "dft"},
    "any": {"in_domain", "dft", "high_level", "experimental", "none"},
}

# Physicality score below which the finder warns about a model
_PHYSICALITY_WARN = 0.5

# Minimum admissible benchmarks for a model to be ranked by the finder
MIN_FINDER_BENCH = 3

# Pseudo-observations shrinking each model's benchmark mean towards the pooled mean,
# so a model that aced a few benchmarks cannot outrank one solid across many
_FINDER_PRIOR_STRENGTH = 4.0

# Weight of the benchmark mean in the finder's fit when elements are selected
FINDER_BENCH_WEIGHT = 0.6

# How many structures the failure ranking keeps by default.
FAILS_TOP = 15

# Most structures one benchmark may place in a field-wide failure ranking
FAILS_PER_BENCHMARK = 2


# --- Shared filters ------------------------------------------------------------------


def categories(rows: list[dict[str, Any]]) -> list[str]:
    """
    List the distinct benchmark categories present, sorted.

    Parameters
    ----------
    rows
        Hardness rows.

    Returns
    -------
    list[str]
        Sorted unique category names.
    """
    return sorted({row["category"] for row in rows})


def element_options(rows: list[dict[str, Any]]) -> list[str]:
    """
    List the distinct element symbols present across all structures, sorted.

    Parameters
    ----------
    rows
        Hardness rows.

    Returns
    -------
    list[str]
        Sorted unique element symbols (empty element sets contribute nothing).
    """
    elements: set[str] = set()
    for row in rows:
        elements.update(row.get("elements") or [])
    return sorted(elements)


def _filter_rows(
    rows: list[dict[str, Any]],
    wanted_categories: list[str] | None,
    elements: list[str] | None,
) -> list[dict[str, Any]]:
    """
    Filter hardness rows by category and chemistry.

    Parameters
    ----------
    rows
        Hardness rows.
    wanted_categories
        Keep only these categories (empty/None = all).
    elements
        Keep only structures containing at least one of these elements (empty/None =
        all). Structures with an unknown element set are excluded once a filter is set.

    Returns
    -------
    list[dict[str, Any]]
        The filtered rows.
    """
    out = rows
    if wanted_categories:
        allowed = set(wanted_categories)
        out = [row for row in out if row["category"] in allowed]
    if elements:
        wanted = set(elements)
        out = [row for row in out if wanted & set(row.get("elements") or [])]
    return out


# --- Model report card: per-model profile across chemistry + extrapolation ----------


@lru_cache(maxsize=128)
def model_scores(model: str | None) -> list[dict[str, Any]]:
    """
    Return the per-benchmark score rows for one model.

    Memoised, so callers must treat the returned list as read-only.

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    list[dict[str, Any]]
        Rows from :func:`load_model_benchmark_scores` for that model.
    """
    if not model:
        return []
    return [row for row in load_model_benchmark_scores() if row["model"] == model]


def _agg(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    """
    Mean score and count for a set of report rows.

    Parameters
    ----------
    rows
        Report rows.

    Returns
    -------
    tuple[float | None, int]
        ``(mean, n)`` -- mean is ``None`` when empty.
    """
    if not rows:
        return None, 0
    return sum(row["score"] for row in rows) / len(rows), len(rows)


def report_card_summary(model: str | None) -> dict[str, Any]:
    """
    Summarise a model's accuracy in-domain vs extrapolated, plus physicality.

    Accuracy benchmarks are split by whether the model's training level of theory
    matches the reference. Physicality is reported separately.

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    dict[str, Any]
        ``{model, training_lot, n_benchmarks, in_domain, extrapolated,
        extrapolated_kinds, physicality}`` where ``in_domain``, ``extrapolated`` and
        ``physicality`` are ``(mean, n)`` and ``extrapolated_kinds`` counts the
        extrapolation classes pooled into the extrapolated mean.
    """
    rows = model_scores(model)
    accuracy = [
        r for r in rows if r["lot_class"] != "none" and r["domain"] != "physicality"
    ]
    in_domain = [r for r in accuracy if r["lot_class"] == "in_domain"]
    extrapolated = [r for r in accuracy if r["lot_class"] != "in_domain"]
    kinds = Counter(r["lot_class"] for r in extrapolated)
    physicality = [r for r in rows if r["domain"] == "physicality"]
    return {
        "model": model,
        "training_lot": rows[0]["model_lot"] if rows else None,
        "n_benchmarks": len(rows),
        "in_domain": _agg(in_domain),
        "extrapolated": _agg(extrapolated),
        "extrapolated_kinds": dict(kinds),
        "physicality": _agg(physicality),
    }


def category_aggregates(model: str | None) -> list[dict[str, Any]]:
    """
    Aggregate a model's scores to one row per category (mean + LoT class breakdown).

    The category's ``lot_class`` is the most common class among its benchmarks, with
    ties broken towards in-domain.

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    list[dict[str, Any]]
        ``{category, domain, mean, lot_class, lot_breakdown, n}`` per category the
        model covers; ``lot_breakdown`` is a hover-ready string like
        ``"in-domain 8/9 · functional mismatch 1/9"``.
    """
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_scores(model):
        by_cat[row["category"]].append(row)
    out: list[dict[str, Any]] = []
    # Classes are validated by the data layer, so LOT_CLASS_STYLE's insertion order
    # (in-domain first) can serve as the tie-break order directly.
    severity = list(LOT_CLASS_STYLE)
    for category, rows in by_cat.items():
        counts = Counter(r["lot_class"] for r in rows)
        dominant = max(counts, key=lambda cls: (counts[cls], -severity.index(cls)))
        breakdown = " · ".join(
            f"{LOT_CLASS_STYLE[cls]['label']} {count}/{len(rows)}"
            for cls, count in counts.most_common()
        )
        out.append(
            {
                "category": category,
                "domain": rows[0]["domain"],
                "mean": sum(r["score"] for r in rows) / len(rows),
                "lot_class": dominant,
                "lot_breakdown": breakdown,
                "n": len(rows),
            }
        )
    return out


@lru_cache(maxsize=1)
def category_model_means() -> dict[str, dict[str, float]]:
    """
    Mean score per (category, model) across every model's covered benchmarks.

    Returns
    -------
    dict[str, dict[str, float]]
        ``category -> {model: mean score}``.
    """
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in load_model_benchmark_scores():
        sums[row["category"]][row["model"]].append(row["score"])
    return {
        category: {model: sum(scores) / len(scores) for model, scores in models.items()}
        for category, models in sums.items()
    }


def ordinal(n: int) -> str:
    """
    Format ``n`` as an English ordinal ("1st", "2nd", "3rd", "4th"...).

    Parameters
    ----------
    n
        A positive integer rank.

    Returns
    -------
    str
        The ordinal string.
    """
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def representative_models(names: list[str]) -> list[str]:
    """
    Reduce a list of model variants to one representative per architecture family.

    Keeps the broadest-coverage variant of each family.

    Parameters
    ----------
    names
        Model display names.

    Returns
    -------
    list[str]
        One model per family, in the input order of first appearance.
    """
    variants = model_taxonomy()["variants"]
    coverage = model_coverage()
    by_family: dict[str, list[str]] = defaultdict(list)
    for name in names:
        family = variants.get(name, {}).get("family", name)
        by_family[family].append(name)
    reps = {
        max(models, key=lambda m: coverage.get(m, 0)) for models in by_family.values()
    }
    return [name for name in names if name in reps]


def report_default_model() -> str | None:
    """
    Choose the report card's default model.

    Prefers the broadest-coverage model with at least one in-domain test, so the
    opening view shows the in-domain vs extrapolated contrast.

    Returns
    -------
    str | None
        Model display name, or ``None`` if there are no scores.
    """
    coverage = model_coverage()
    if not coverage:
        return None
    has_in_domain = {
        row["model"]
        for row in load_model_benchmark_scores()
        if row["lot_class"] == "in_domain"
    }
    candidates = has_in_domain or set(coverage)
    return max(candidates, key=lambda m: coverage[m])


def model_options() -> list[dict[str, str]]:
    """
    List the options for a single searchable model picker.

    Returns
    -------
    list[dict[str, str]]
        One option per model, family-sorted.
    """
    taxonomy = model_taxonomy()
    variants = taxonomy["variants"]
    options: list[dict[str, str]] = []
    for family_meta in taxonomy["families"].values():
        for m in family_meta["models"]:
            lot = variants.get(m, {}).get("training_lot") or "?"
            options.append(
                {"label": f"{family_meta['label']}  ·  {m}  ·  {lot}", "value": m}
            )
    return options


def default_h2h_models(
    models: list[str], error_matrix: dict[str, dict[str, float]]
) -> tuple[str | None, str | None]:
    """
    Pick the two models evaluated on the most structures as the default pair.

    The first two models alphabetically may share no structures, so pick by structure
    coverage instead.

    Parameters
    ----------
    models
        Model names.
    error_matrix
        The per-structure error matrix.

    Returns
    -------
    tuple[str | None, str | None]
        The two highest-coverage model names (or ``None`` where unavailable).
    """
    coverage: dict[str, int] = defaultdict(int)
    for resids in error_matrix.values():
        for model in resids:
            coverage[model] += 1
    ranked = sorted(models, key=lambda m: coverage.get(m, 0), reverse=True)
    first = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    return first, second


# --- Head-to-head: two models, structure by structure ---------------------------------


def head_to_head_stats(
    model_a: str, model_b: str, error_matrix: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """
    Compare two models structure-by-structure over the structures they both cover.

    A model wins a structure when its absolute error there is smaller. Each benchmark
    contributes one win rate, a category averages its benchmarks, and the overall
    verdict averages the categories, so one large benchmark cannot dominate. Ties are
    excluded from every denominator, so ``win_rate_a + win_rate_b == 1``.

    Parameters
    ----------
    model_a, model_b
        Model names to compare.
    error_matrix
        ``structure_key -> {model: signed_residual}`` from :func:`load_error_matrix`.

    Returns
    -------
    dict[str, Any]
        ``{"n_shared", "wins_a", "wins_b", "ties", "overall_win_rate_a",
        "per_category"}``. Each ``per_category`` row is ``{"category", "n", "wins_a",
        "wins_b", "ties", "decisive", "n_benchmarks", "win_rate_a"}`` where
        ``win_rate_a`` is the benchmark-averaged rate and ``n`` the structure count
        (rows sorted by descending structure count).
    """
    # Per benchmark (category/benchmark/metric): [wins_a, wins_b, ties].
    per_bench: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    bench_category: dict[str, str] = {}
    for key, resids in error_matrix.items():
        if model_a not in resids or model_b not in resids:
            continue
        ea, eb = abs(resids[model_a]), abs(resids[model_b])
        category, bench, _ = split_structure_key(key)
        bench_category[bench] = category
        bucket = per_bench[bench]
        if ea < eb:
            bucket[0] += 1
        elif eb < ea:
            bucket[1] += 1
        else:
            bucket[2] += 1

    # Roll benchmarks up into categories: each benchmark's win rate counts once.
    cat_bench_rates: dict[str, list[float]] = defaultdict(list)
    cat_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # a, b, ties
    for bench, (win_a, win_b, tie) in per_bench.items():
        category = bench_category[bench]
        decisive = win_a + win_b
        if decisive:
            cat_bench_rates[category].append(win_a / decisive)
        counts = cat_counts[category]
        counts[0] += win_a
        counts[1] += win_b
        counts[2] += tie

    per_category = []
    for category, (win_a, win_b, tie) in cat_counts.items():
        rates = cat_bench_rates.get(category, [])
        per_category.append(
            {
                "category": category,
                "n": win_a + win_b + tie,  # structures compared (incl. ties)
                "wins_a": win_a,
                "wins_b": win_b,
                "ties": tie,
                "decisive": win_a + win_b,  # for the Wilson whisker
                "n_benchmarks": len(rates),
                # Equal-weight the benchmarks; an all-tie category reads even.
                "win_rate_a": sum(rates) / len(rates) if rates else 0.5,
            }
        )
    per_category.sort(key=lambda row: row["n"], reverse=True)

    # Overall verdict: average the category win rates (each area counts equally).
    cat_rates = [row["win_rate_a"] for row in per_category if row["decisive"]]
    overall_win_rate_a = sum(cat_rates) / len(cat_rates) if cat_rates else 0.5

    wins_a = sum(bucket[0] for bucket in per_bench.values())
    wins_b = sum(bucket[1] for bucket in per_bench.values())
    ties = sum(bucket[2] for bucket in per_bench.values())
    return {
        "n_shared": wins_a + wins_b + ties,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "overall_win_rate_a": overall_win_rate_a,
        "per_category": per_category,
    }


@lru_cache(maxsize=128)
def cached_head_to_head_stats(model_a: str, model_b: str) -> dict[str, Any]:
    """
    Memoised :func:`head_to_head_stats` over the process-stable error matrix.

    Parameters
    ----------
    model_a, model_b
        Model names to compare.

    Returns
    -------
    dict[str, Any]
        The head-to-head stats (read-only; see :func:`head_to_head_stats`).
    """
    return head_to_head_stats(model_a, model_b, load_error_matrix())


def h2h_pair_stats(model_a: str | None, model_b: str | None) -> dict[str, Any] | None:
    """
    Return the cached head-to-head stats for a model pair, or ``None`` if not a pair.

    Parameters
    ----------
    model_a, model_b
        The selected models (either may be unset, or both the same).

    Returns
    -------
    dict[str, Any] | None
        The stats for two distinct models, else ``None``.
    """
    if not model_a or not model_b or model_a == model_b:
        return None
    return cached_head_to_head_stats(model_a, model_b)


def h2h_area_options(stats: dict[str, Any] | None) -> list[dict[str, str]]:
    """
    List the areas a model pair shares, as options for the drilldown picker.

    Parameters
    ----------
    stats
        Pair stats from :func:`h2h_pair_stats`, or ``None`` when not a valid pair.

    Returns
    -------
    list[dict[str, str]]
        One ``{"label", "value"}`` option per shared category, sorted by name.
    """
    names = sorted(row["category"] for row in (stats or {}).get("per_category", []))
    return [{"label": name, "value": name} for name in names]


def h2h_verdict_counts(stats: dict[str, Any]) -> tuple[int, int, int]:
    """
    Count the categories each model leads in, and the practical ties.

    Parameters
    ----------
    stats
        Head-to-head stats from :func:`head_to_head_stats`.

    Returns
    -------
    tuple[int, int, int]
        ``(led_by_a, led_by_b, too_close)`` using :data:`H2H_TIE_BAND`, so the headline
        agrees with the bar colours by construction.
    """
    led_a = sum(
        1 for c in stats["per_category"] if c["win_rate_a"] - 0.5 > H2H_TIE_BAND
    )
    led_b = sum(
        1 for c in stats["per_category"] if c["win_rate_a"] - 0.5 < -H2H_TIE_BAND
    )
    return led_a, led_b, len(stats["per_category"]) - led_a - led_b


def benchmark_error_medians(
    error_matrix: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Median absolute error per benchmark, for unit-free head-to-head normalisation.

    Dividing by this makes errors from benchmarks with different units comparable.

    Parameters
    ----------
    error_matrix
        The per-structure error matrix.

    Returns
    -------
    dict[str, float]
        Mapping ``category/benchmark/metric -> median |error|`` (over all models and
        structures in that benchmark).
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for key, resids in error_matrix.items():
        _, bench, _ = split_structure_key(key)
        for resid in resids.values():
            groups[bench].append(abs(resid))
    return {
        bench: statistics.median(values) for bench, values in groups.items() if values
    }


@lru_cache(maxsize=1)
def cached_benchmark_error_medians() -> dict[str, float]:
    """
    Memoised :func:`benchmark_error_medians` over the process-stable error matrix.

    Returns
    -------
    dict[str, float]
        ``category/benchmark/metric -> median |error|``.
    """
    return benchmark_error_medians(load_error_matrix())


# --- Model finder: "which model for MY system?" --------------------------------------


def model_finder_rankings(
    wanted_categories: list[str] | None = None,
    elements: list[str] | None = None,
    tolerance: str = "any",
) -> list[dict[str, Any]]:
    """
    Rank models for a user-described use case, with the evidence behind each.

    Combines the model's mean score over admissible benchmarks, shrunk towards the
    pooled mean by coverage, with its mean error rank on structures containing the
    selected elements. Models with fewer than :data:`MIN_FINDER_BENCH` admissible
    benchmarks are dropped.

    Parameters
    ----------
    wanted_categories
        Benchmark categories to restrict to (empty/None = all).
    elements
        Element symbols the user's system contains (empty/None = skip structure
        evidence).
    tolerance
        One of the ``FINDER_TOLERANCE_OPTIONS`` values -- how much extrapolation the
        benchmark evidence may include.

    Returns
    -------
    list[dict[str, Any]]
        Ranked rows ``{model, fit, bench_mean, n_bench, in_domain_frac, struct_score,
        n_struct, no_element_evidence, physicality, low_physicality, training_lot}``,
        best first.
    """
    admissible = _TOLERANCE_CLASSES.get(tolerance, _TOLERANCE_CLASSES["any"])
    wanted_cats = set(wanted_categories or [])
    bench: dict[str, list[float]] = defaultdict(list)
    in_domain_n: dict[str, int] = defaultdict(int)
    phys: dict[str, list[float]] = defaultdict(list)
    lots: dict[str, str | None] = {}
    for row in load_model_benchmark_scores():
        if lots.get(row["model"]) is None:
            lots[row["model"]] = row["model_lot"]
        # Physicality applies whatever the selection
        if row["domain"] == "physicality":
            phys[row["model"]].append(row["score"])
            continue
        if wanted_cats and row["category"] not in wanted_cats:
            continue
        if row["lot_class"] not in admissible:
            continue
        bench[row["model"]].append(row["score"])
        if row["lot_class"] == "in_domain":
            in_domain_n[row["model"]] += 1
    candidates = {
        model: scores
        for model, scores in bench.items()
        if len(scores) >= MIN_FINDER_BENCH
    }
    if not candidates:
        return []
    pooled = [score for scores in candidates.values() for score in scores]
    prior = sum(pooled) / len(pooled)

    # model -> benchmark -> standings, so each benchmark counts once
    standings: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if elements:
        ranks = structure_model_ranks()
        for row in _filter_rows(load_hardness(), wanted_categories, elements):
            bench = (row["category"], row["benchmark"])
            for model, rank in ranks.get(row["key"], {}).items():
                standings[model][bench].append(1.0 - rank)

    results: list[dict[str, Any]] = []
    for model, scores in candidates.items():
        bench_mean = sum(scores) / len(scores)
        bench_adj = (len(scores) * bench_mean + _FINDER_PRIOR_STRENGTH * prior) / (
            len(scores) + _FINDER_PRIOR_STRENGTH
        )
        per_bench = [sum(v) / len(v) for v in standings.get(model, {}).values()]
        struct_score = sum(per_bench) / len(per_bench) if per_bench else None
        n_struct = sum(len(v) for v in standings.get(model, {}).values())
        if elements and struct_score is not None:
            fit = FINDER_BENCH_WEIGHT * bench_adj + (1 - FINDER_BENCH_WEIGHT) * (
                struct_score
            )
        else:
            fit = bench_adj
        phys_scores = phys.get(model)
        physicality = sum(phys_scores) / len(phys_scores) if phys_scores else None
        results.append(
            {
                "model": model,
                "fit": fit,
                "bench_mean": bench_mean,
                "n_bench": len(scores),
                "in_domain_frac": in_domain_n[model] / len(scores),
                "struct_score": struct_score,
                "n_struct": n_struct,
                "no_element_evidence": bool(elements) and not n_struct,
                "physicality": physicality,
                "low_physicality": (
                    physicality is not None and physicality < _PHYSICALITY_WARN
                ),
                "training_lot": lots.get(model),
            }
        )
    results.sort(key=lambda r: r["fit"], reverse=True)
    return results


@lru_cache(maxsize=64)
def cached_finder_rankings(
    wanted_categories: tuple[str, ...] | None,
    elements: tuple[str, ...] | None,
    tolerance: str = "any",
) -> list[dict[str, Any]]:
    """
    Memoised :func:`model_finder_rankings` keyed on hashable use-case controls.

    Parameters
    ----------
    wanted_categories, elements
        Selections as hashable tuples (``None`` for "all"/"skip").
    tolerance
        Extrapolation tolerance (see :func:`model_finder_rankings`).

    Returns
    -------
    list[dict[str, Any]]
        The ranked rows (read-only).
    """
    return model_finder_rankings(
        wanted_categories=list(wanted_categories) if wanted_categories else None,
        elements=list(elements) if elements else None,
        tolerance=tolerance,
    )


# --- Where models fail: the hardest structures ---------------------------------------


def structure_fails_ranking(
    rows: list[dict[str, Any]],
    wanted_categories: list[str] | None,
    model: str | None,
    sort: str,
    top: int = FAILS_TOP,
) -> list[dict[str, Any]]:
    """
    Rank the structures models fail on hardest, or where one model fails worst.

    With no model, ranks field-wide by the share of models for which the structure is
    an outlier (``"hardest"``, ties broken by difficulty percentile) or by model
    disagreement (``"contested"``), both unit-free, keeping at most
    :data:`FAILS_PER_BENCHMARK` structures per benchmark. With a model, ranks by that
    model's error over the median model's error on each structure.

    Parameters
    ----------
    rows
        Hardness rows from :func:`load_hardness`.
    wanted_categories
        Restrict to these benchmark categories (empty/None = all).
    model
        If set, rank by this model's per-structure error rank instead of difficulty.
    sort
        ``"hardest"`` or ``"contested"`` (ignored when ``model`` is set).
    top
        Keep at most this many structures.

    Returns
    -------
    list[dict[str, Any]]
        The worst-first structures, each a hardness row plus ``fail_score`` (the
        ranking value) and, in per-model mode, ``model_rank`` (0 = best of the cohort,
        1 = worst) and ``model_err``.
    """
    filtered = _filter_rows(rows, wanted_categories, None)
    if model:
        ranks = structure_model_ranks()
        errors = load_error_matrix()
        scored: list[dict[str, Any]] = []
        for row in filtered:
            rank = ranks.get(row["key"], {}).get(model)
            if rank is None:  # model was not evaluated on this structure
                continue
            resids = errors.get(row["key"], {})
            resid = resids.get(model)
            # Same-precision numerator and denominator, both from the error matrix
            median = statistics.median(abs(v) for v in resids.values())
            if resid is None or not median:
                continue
            scored.append(
                {
                    **row,
                    "fail_score": abs(resid) / median,
                    "model_rank": rank,
                    "model_err": abs(resid),
                }
            )
        scored.sort(key=lambda r: (r["fail_score"], r["model_rank"]), reverse=True)
        return scored[:top]
    key = "disagreement" if sort == "contested" else "fail_frac"
    # "Hardest" lists only structures at least one model fails
    ranked = [
        row
        for row in filtered
        if row.get(key) is not None and (sort == "contested" or row[key] > 0)
    ]
    ranked.sort(key=lambda r: (-r[key], -r["hardness_pct"], r["key"]))
    kept: list[dict[str, Any]] = []
    per_bench: dict[tuple[str, str], int] = defaultdict(int)
    for row in ranked:
        if len(kept) >= top:
            break
        bench = (row["category"], row["benchmark"])
        if per_bench[bench] >= FAILS_PER_BENCHMARK:
            continue
        per_bench[bench] += 1
        kept.append({**row, "fail_score": row[key]})
    return kept


@lru_cache(maxsize=1)
def offered_categories() -> frozenset[str]:
    """
    Return the categories the Explorer's area pickers offer.

    Returns
    -------
    frozenset[str]
        Categories with at least one structure.
    """
    return frozenset(categories(load_hardness()))


@lru_cache(maxsize=128)
def cached_structure_fails_ranking(
    wanted_categories: tuple[str, ...] | None,
    model: str | None,
    sort: str,
    top: int = FAILS_TOP,
) -> list[dict[str, Any]]:
    """
    Memoised :func:`structure_fails_ranking` over the process-stable hardness rows.

    Parameters
    ----------
    wanted_categories
        Benchmark areas as a hashable tuple (``None`` for all).
    model
        Focus model, or ``None`` for the field-wide ranking.
    sort
        ``"hardest"`` or ``"contested"`` (ignored when a model is focused).
    top
        Maximum rows to return.

    Returns
    -------
    list[dict[str, Any]]
        The ranked rows (read-only; see :func:`structure_fails_ranking`).
    """
    return structure_fails_ranking(
        load_hardness(),
        list(wanted_categories) if wanted_categories else None,
        model,
        sort,
        top,
    )
