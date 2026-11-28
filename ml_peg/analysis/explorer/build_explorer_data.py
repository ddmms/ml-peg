"""
Derive the cross-model Explorer signals from the shipped benchmark data.

Rebuilds a ``structure x model`` error matrix from the parity figures in
``ml_peg/app/data``, then derives per-structure hardness rows and a model
error-correlation matrix. A parity residual is the prediction minus the reference.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from functools import cache
import json
import logging
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import orjson
from scipy.cluster.hierarchy import leaves_list, linkage

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "app" / "data"

# Models a structure needs for its hardness and disagreement to be meaningful
MIN_MODELS = 3

# A model fails a structure when its error exceeds this multiple of its own median
# error on the benchmark (a per-model outlier test, so units cancel)
OUTLIER_FACTOR = 3.0

# Shared structures a model pair needs before its error correlation is reported
MIN_SHARED_STRUCTURES = 10

# Benchmarks that measure cost rather than accuracy, left out of every Explorer view.
EXCLUDED_BENCHMARKS = frozenset({("molecular_dynamics", "inference_speed")})

_ERROR_DECIMALS = 6
_RATIO_DECIMALS = 4


def data_root() -> Path:
    """
    Return the directory the Explorer reads shipped benchmark data from.

    Reading through this, not an imported :data:`DATA_ROOT`, lets tests redirect it.

    Returns
    -------
    Path
        Directory holding one subdirectory per category.
    """
    return DATA_ROOT


class DataFileIndex(NamedTuple):
    """The benchmark-level data files found by one walk of :func:`data_root`."""

    figures: tuple[Path, ...]
    """Sorted paths of every benchmark-level ``figure_*.json``."""
    metrics_tables: tuple[Path, ...]
    """Sorted paths of every ``*_metrics_table.json``."""
    model_dirs: dict[tuple[str, str], tuple[Path, ...]]
    """``(category, benchmark)`` -> sorted per-model output directories."""


@cache
def data_file_index() -> DataFileIndex:
    """
    Locate every benchmark-level data file with a single pruned walk of the data tree.

    The walk stops below each benchmark directory, which avoids reading the per-model
    geometry files.

    Returns
    -------
    DataFileIndex
        The figures, metrics tables, and per-benchmark model directories.
    """
    figures: list[Path] = []
    tables: list[Path] = []
    model_dirs: dict[tuple[str, str], tuple[Path, ...]] = {}
    root = data_root()
    pending = [(root, frozenset())]
    while pending:
        here, ancestors = pending.pop()
        resolved = here.resolve()
        if resolved in ancestors:
            continue
        children = sorted(here.iterdir()) if here.is_dir() else []
        dirs = [p for p in children if p.is_dir()]
        filenames = [p.name for p in children if p.is_file()]
        rel = here.relative_to(root).parts
        key = (rel[0], "/".join(rel[1:])) if len(rel) >= 2 else None
        if key in EXCLUDED_BENCHMARKS:
            continue
        figs = [f for f in filenames if f.startswith("figure_") and f.endswith(".json")]
        tabs = [f for f in filenames if f.endswith("_metrics_table.json")]
        figures.extend(here / f for f in figs)
        tables.extend(here / f for f in tabs)
        if figs or tabs:
            if key:
                model_dirs[key] = tuple(dirs)
        else:
            pending.extend((directory, ancestors | {resolved}) for directory in dirs)
    return DataFileIndex(tuple(sorted(figures)), tuple(sorted(tables)), model_dirs)


def read_json(path: Path) -> Any:
    """
    Parse a JSON file with orjson, falling back to the stdlib for bare ``NaN`` tokens.

    Mirrors ``ml_peg.app.utils.load.read_json`` so this module does not import Dash.

    Parameters
    ----------
    path
        Path to the JSON file.

    Returns
    -------
    Any
        The parsed document.
    """
    data = path.read_bytes()
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError:
        return json.loads(data)


# Column positions follow the corresponding analysis producers' hoverdata.
_OBSERVATION_COLUMNS = {
    ("BH9", "bh9_barriers"): (2,),
    ("BH2O_36", "bh2o_36_barriers"): (0, 1),
    ("GMTKN55", "rel_energies"): (0, 2),
    ("Wiggle150", "relative_energies"): (0, 1),
    ("isomer_complexes", "isomer_complexes"): (0, 1),
}
_NO_GEOMETRY = {"BH2O_36", "GMTKN55", "Wiggle150"}


def _observation_id(customdata: Any, columns: tuple[int, ...]) -> str | None:
    """
    Read one observation identifier from the benchmark's hover columns.

    Parameters
    ----------
    customdata
        Scalar or row of hover metadata.
    columns
        Positions that jointly identify the observation.

    Returns
    -------
    str or None
        Observation identifier, if all required fields are present.
    """
    values = customdata if isinstance(customdata, list) else [customdata]
    if any(
        i >= len(values)
        or type(values[i]) not in (str, int)
        or not str(values[i]).strip()
        or "::" in str(values[i])
        for i in columns
    ):
        return None
    if len(columns) == 1:
        return str(values[columns[0]])
    return json.dumps(
        [values[i] for i in columns], ensure_ascii=False, separators=(",", ":")
    )


def _iter_parity_traces(
    figure: dict[str, Any], benchmark: str = "", metric: str = ""
) -> Iterator[tuple[str, list[tuple[str, float | None, float | None]]]]:
    """
    Yield each model's observation IDs, references and signed residuals.

    Parameters
    ----------
    figure
        Saved Plotly figure.
    benchmark
        Benchmark directory, including any parent directories.
    metric
        Metric identified by the figure filename.

    Yields
    ------
    tuple
        Model name and points containing ID, reference and residual. Missing
        values retain their IDs so duplicate observations can still be detected.
    """
    schema = (benchmark.rsplit("/", 1)[-1], metric)
    columns = _OBSERVATION_COLUMNS.get(schema, (0,))
    for trace in figure.get("data", []):
        if "markers" not in (trace.get("mode") or ""):
            continue
        hover = trace.get("hovertemplate") or ""
        if "<b>Pred: </b>%{x}" in hover and "<b>Ref: </b>%{y}" in hover:
            x, y = trace.get("x"), trace.get("y")
        elif "<b>Pred: </b>%{y}" in hover and "<b>Ref: </b>%{x}" in hover:
            x, y = trace.get("y"), trace.get("x")
        else:
            continue
        model = trace.get("name")
        customdata = trace.get("customdata")
        if (
            not isinstance(model, str)
            or not model
            or not all(isinstance(values, list) for values in (customdata, x, y))
        ):
            continue
        if len(customdata) != len(x) or len(x) != len(y):
            continue
        points = []
        for cd, prediction, reference in zip(customdata, x, y, strict=True):
            sid = _observation_id(cd, columns)
            if sid is None or (
                schema == ("GMTKN55", "rel_energies")
                and (len(cd) < 4 or cd[3] is not False)
            ):
                continue
            residual = None
            ref = None
            if type(prediction) in (int, float) and type(reference) in (int, float):
                difference = prediction - reference
                if math.isfinite(difference):
                    ref = reference
                    residual = difference
            points.append((sid, ref, residual))
        if points:
            yield model, points


def _figure_residuals(
    figure: dict[str, Any], benchmark: str, metric: str
) -> dict[str, dict[str, float]]:
    """
    Collect residuals only for unique IDs with a common reference.

    Parameters
    ----------
    figure
        Saved Plotly figure.
    benchmark
        Benchmark directory, including any parent directories.
    metric
        Metric identified by the figure filename.

    Returns
    -------
    dict
        Observation IDs mapped to signed residuals by model.
    """
    observations: dict[str, dict[str, float]] = defaultdict(dict)
    references: dict[str, set[float]] = defaultdict(set)
    counts: Counter[tuple[str, str]] = Counter()
    for model, points in _iter_parity_traces(figure, benchmark, metric):
        for sid, ref, residual in points:
            counts[model, sid] += 1
            if ref is not None and residual is not None:
                references[sid].add(ref)
                observations[sid][model] = residual
    ambiguous = {sid for (_, sid), count in counts.items() if count > 1}
    return {
        sid: residuals
        for sid, residuals in observations.items()
        if sid not in ambiguous and len(references[sid]) == 1
    }


def _benchmark_key(figure_path: Path) -> tuple[str, str, str]:
    """
    Split a ``figure_*.json`` path into ``(category, benchmark, metric)``.

    Parameters
    ----------
    figure_path
        Path to a parity figure under :func:`data_root`.

    Returns
    -------
    tuple[str, str, str]
        Top-level category, benchmark (any intermediate directories joined by "/"),
        and the metric (the figure filename stem without the ``figure_`` prefix).
    """
    rel = figure_path.relative_to(data_root())
    parts = rel.parts
    category = parts[0]
    benchmark = "/".join(parts[1:-1])
    metric = figure_path.stem.removeprefix("figure_")
    return category, benchmark, metric


def _xyz_element_union(path: Path) -> list[str] | None:
    """
    Read the set of element symbols in an ``.xyz``/``.extxyz`` file (all frames).

    Parses only the first token of each atom line, avoiding a full ASE read.

    Parameters
    ----------
    path
        Path to a shipped geometry file.

    Returns
    -------
    list[str] | None
        Sorted unique element symbols, or ``None`` if the file cannot be read or holds
        no recognisable symbols (e.g. atomic-number-indexed geometries).
    """
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        logger.warning("Cannot read geometry %s: %s", path, exc)
        return None
    symbols: set[str] = set()
    i, n = 0, len(lines)
    while i < n:
        header = lines[i].strip()
        if not header:
            i += 1
            continue
        try:
            n_atoms = int(header)
        except ValueError:
            # Not a frame header (stray/blank line); skip and resync.
            i += 1
            continue
        # header, then a comment line, then n_atoms atom lines "<symbol> x y z ...".
        for k in range(i + 2, min(i + 2 + n_atoms, n)):
            tokens = lines[k].split()
            if tokens and tokens[0].isalpha():
                symbols.add(tokens[0])
        i = i + 2 + n_atoms
    return sorted(symbols) if symbols else None


@cache
def _benchmark_geometry_index(
    category: str, benchmark: str
) -> tuple[dict[str, Path | None], dict[str, Path | None]]:
    """
    Index a benchmark's shipped ``.xyz`` geometries by structure id.

    Scans the first model directory, as geometries are shared across models. Each
    ``_``-delimited tail of a stem is also indexed, since some ids are a suffix of the
    file name (``AE11_1`` is shipped as ``AE11_AE11_1.xyz``).

    Parameters
    ----------
    category
        Benchmark category directory.
    benchmark
        Benchmark directory (relative to the category).

    Returns
    -------
    tuple[dict[str, Path], dict[str, Path]]
        ``(exact, suffix)`` maps from structure id to geometry path; ``exact`` wins.
    """
    exact: dict[str, Path | None] = {}
    suffix: dict[str, Path | None] = {}
    model_dirs = data_file_index().model_dirs.get((category, benchmark), ())
    if not model_dirs:
        return exact, suffix
    for file in sorted(model_dirs[0].rglob("*.xyz")):
        stem = file.stem
        exact[stem] = None if stem in exact else file
        pos = stem.find("_")
        while pos != -1:
            tail = stem[pos + 1 :]
            suffix[tail] = None if tail in suffix else file
            pos = stem.find("_", pos + 1)
    return exact, suffix


@cache
def _benchmark_element_index(category: str, benchmark: str) -> dict[str, list[str]]:
    """
    Map a benchmark's geometry-file stems to their element sets via ``info.json``.

    Only trusted when ``elements`` and ``filenames`` are aligned per structure;
    otherwise the geometry files are parsed instead.

    Parameters
    ----------
    category
        Benchmark category directory.
    benchmark
        Benchmark directory (relative to the category).

    Returns
    -------
    dict[str, list[str]]
        ``geometry stem -> sorted element symbols``; empty when ``info.json`` is
        absent or not per-structure.
    """
    info_path = data_root() / category / benchmark / "info.json"
    try:
        payload = read_json(info_path)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    elements = payload.get("elements")
    filenames = payload.get("filenames")
    if (
        not isinstance(elements, list)
        or not isinstance(filenames, list)
        or len(elements) != len(filenames)
        or not all(isinstance(els, list) and els for els in elements)
    ):
        return {}
    return {
        str(stem): sorted({str(symbol) for symbol in els})
        for stem, els in zip(filenames, elements, strict=True)
    }


def _resolve_structure(
    category: str, benchmark: str, sid: str, metric: str = ""
) -> tuple[str | None, list[str]]:
    """
    Resolve a structure's shipped geometry path and element set from the shipped data.

    Parameters
    ----------
    category
        Benchmark category directory.
    benchmark
        Benchmark directory (relative to the category).
    sid
        Observation id from the parity figure.
    metric
        Metric from the parity figure.

    Returns
    -------
    tuple[str | None, list[str]]
        The geometry path relative to :func:`data_root` (POSIX style, for the WEAS
        viewer) or ``None`` if no geometry is shipped, and the sorted element set (empty
        when it cannot be recovered).
    """
    exact, suffix = _benchmark_geometry_index(category, benchmark)
    name = benchmark.rsplit("/", 1)[-1]
    if name in _NO_GEOMETRY:
        return None, []
    if (name, metric) == ("isomer_complexes", "isomer_complexes"):
        sid = "_".join(map(str, json.loads(sid)))
    file = exact.get(sid, suffix.get(sid))
    if file is None:
        return None, []
    elements = _benchmark_element_index(category, benchmark).get(file.stem)
    if elements is None:
        elements = _xyz_element_union(file) or []
    return file.relative_to(data_root()).as_posix(), elements


def compute_explorer_data() -> dict[str, Any]:
    """
    Build the Explorer signals in memory from the shipped benchmark data.

    Returns
    -------
    dict[str, Any]
        ``{"hardness", "error_matrix", "similarity"}`` -- the hardness rows, the
        ``structure_key -> {model: residual}`` matrix, and the similarity payload.
    """
    figures = data_file_index().figures

    structures: dict[str, dict[str, Any]] = {}
    per_benchmark: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    all_models: set[str] = set()

    skipped = 0
    for figure_path in figures:
        try:
            figure = read_json(figure_path)
            category, benchmark, metric = _benchmark_key(figure_path)
            bench_id = f"{category}/{benchmark}/{metric}"

            # OC157 compositions repeat across systems.
            if benchmark == "OC157" and metric == "rel_energies":
                continue
            for sid, resids in _figure_residuals(figure, benchmark, metric).items():
                key = f"{bench_id}::{sid}"
                xyz, elements = _resolve_structure(category, benchmark, sid, metric)
                structures[key] = {
                    "key": key,
                    "sid": sid,
                    "category": category,
                    "benchmark": benchmark,
                    "metric": metric,
                    "elements": elements,
                    "xyz": xyz,
                    "resids": resids,
                }
                for model, resid in resids.items():
                    all_models.add(model)
                    per_benchmark[bench_id][model][key] = resid
        except (ValueError, TypeError, OSError, AttributeError) as exc:
            skipped += 1
            logger.warning(
                "Skipping malformed Explorer figure %s: %s", figure_path, exc
            )
            continue
    if skipped:
        logger.warning(
            "Explorer data build skipped %d of %d parity figures", skipped, len(figures)
        )

    hardness_rows, error_matrix = _summarise_structures(structures, per_benchmark)
    similarity = _model_similarity(per_benchmark, sorted(all_models))
    return {
        "hardness": hardness_rows,
        "error_matrix": error_matrix,
        "similarity": similarity,
    }


def _summarise_structures(
    structures: dict[str, dict[str, Any]],
    per_benchmark: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """
    Reduce each structure's per-model errors to hardness and disagreement signals.

    Parameters
    ----------
    structures
        Mapping from structure key to its accumulated row (including ``resids``).
    per_benchmark
        ``benchmark -> model -> {structure_key: error}``, for each model's typical
        error on a benchmark. A model whose typical error is zero is left out of
        ``fail_frac``, which is ``None`` when no model can be judged.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, dict[str, float]]]
        Hardness rows without per-model errors, so the plot payload stays small, and
        the ``structure_key -> {model: signed_residual}`` error matrix. Structures with
        fewer than two models are dropped from these views.
    """
    typical = {
        (bench, model): float(np.median(np.abs(list(errs.values()))))
        for bench, model_errs in per_benchmark.items()
        for model, errs in model_errs.items()
    }
    rows: list[dict[str, Any]] = []
    error_matrix: dict[str, dict[str, float]] = {}
    for row in structures.values():
        resids = row.pop("resids")
        if len(resids) < 2:
            continue
        abs_by_model = {m: abs(v) for m, v in resids.items()}
        magnitudes = np.array(list(abs_by_model.values()), dtype=float)
        median = float(np.median(magnitudes))
        bench = f"{row['category']}/{row['benchmark']}/{row['metric']}"
        judged = [
            err > OUTLIER_FACTOR * typical[(bench, m)]
            for m, err in abs_by_model.items()
            if typical[(bench, m)] > 0
        ]
        row["n_failing"], row["n_judged"] = sum(judged), len(judged)
        row["fail_frac"] = (
            round(sum(judged) / len(judged), _RATIO_DECIMALS) if judged else None
        )
        mean = float(np.mean(magnitudes))
        std = float(np.std(magnitudes))
        # Coefficient of variation, so disagreement is comparable across units
        disagreement = float(std / mean) if mean > 0 else 0.0
        row["median_err"] = round(median, _ERROR_DECIMALS)
        row["disagreement"] = round(disagreement, _ERROR_DECIMALS)
        row["n_models"] = len(resids)
        row["worst_model"] = max(abs_by_model, key=abs_by_model.get)
        rows.append(row)
        error_matrix[row["key"]] = resids

    _add_hardness_percentile(rows)
    return rows, error_matrix


def _add_hardness_percentile(rows: list[dict[str, Any]]) -> None:
    """
    Annotate each row with ``hardness_pct``: its median-error percentile in-benchmark.

    0 is the easiest structure and 1 the hardest, comparable across benchmarks.

    Parameters
    ----------
    rows
        Hardness rows (mutated in place to add ``hardness_pct``).
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["benchmark"], row["metric"])].append(row)
    for group in groups.values():
        order = sorted(group, key=lambda r: r["median_err"])
        denom = max(len(order) - 1, 1)
        for rank, row in enumerate(order):
            row["hardness_pct"] = round(rank / denom, _RATIO_DECIMALS)


def _model_similarity(
    per_benchmark: dict[str, dict[str, dict[str, float]]],
    models: list[str],
) -> dict[str, Any]:
    """
    Compute the model x model error-correlation matrix and a clustering leaf order.

    Signed residuals are z-scored within each metric, then correlated over shared
    observations. Metrics with more shared observations contribute more points.

    Parameters
    ----------
    per_benchmark
        ``benchmark -> model -> {structure_key: error}``.
    models
        Sorted list of all model names.

    Returns
    -------
    dict[str, Any]
        Model names, clustering order, correlations and pairwise shared counts.
    """
    normalised: dict[str, dict[str, float]] = defaultdict(dict)
    for model_errors in per_benchmark.values():
        for model, key_errs in model_errors.items():
            vals = np.array(list(key_errs.values()), dtype=float)
            if len(vals) < 2:
                continue
            mean = vals.mean()
            std = vals.std()
            if std == 0:
                continue
            for key, err in key_errs.items():
                normalised[key][model] = (err - mean) / std

    n = len(models)
    corr = np.full((n, n), np.nan)
    shared_counts = np.zeros((n, n), dtype=int)
    np.fill_diagonal(corr, 1.0)
    columns: dict[str, dict[str, float]] = defaultdict(dict)
    for key, model_vals in normalised.items():
        for model, z in model_vals.items():
            columns[model][key] = z

    for i, mi in enumerate(models):
        shared_counts[i, i] = len(columns[mi])
        for j in range(i + 1, n):
            mj = models[j]
            shared = columns[mi].keys() & columns[mj].keys()
            shared_counts[i, j] = shared_counts[j, i] = len(shared)
            if len(shared) < MIN_SHARED_STRUCTURES:
                continue
            a = np.array([columns[mi][k] for k in shared])
            b = np.array([columns[mj][k] for k in shared])
            if a.std() == 0 or b.std() == 0:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            corr[i, j] = corr[j, i] = c

    order = _cluster_order(corr)
    corr_out = [
        [None if np.isnan(v) else round(float(v), _RATIO_DECIMALS) for v in row]
        for row in corr
    ]
    return {
        "models": models,
        "order": order,
        "corr": corr_out,
        "shared_counts": shared_counts.tolist(),
    }


def _cluster_order(corr: np.ndarray) -> list[int]:
    """
    Order models by hierarchical clustering of ``1 - correlation`` distances.

    Parameters
    ----------
    corr
        Symmetric model x model correlation matrix (may contain NaNs).

    Returns
    -------
    list[int]
        Leaf order of model indices. Falls back to the identity order when there are
        fewer than three models or the distances are degenerate.
    """
    n = corr.shape[0]
    if n < 3:
        return list(range(n))
    # Pairs with too few shared structures sit at the mean correlation, so coverage
    # does not drive the order
    off_diagonal = corr[~np.eye(n, dtype=bool)]
    defined = off_diagonal[~np.isnan(off_diagonal)]
    fill = float(defined.mean()) if defined.size else 0.0
    dist = 1.0 - np.where(np.isnan(corr), fill, corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = dist[np.triu_indices(n, k=1)]
    if not np.any(condensed):
        return list(range(n))
    linkage_matrix = linkage(condensed, method="average")
    return [int(i) for i in leaves_list(linkage_matrix)]
