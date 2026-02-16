"""Helpers for filtering benchmark data by allowed element groups."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
from yaml import safe_load


def normalize_element_set_key(key: str) -> str:
    """
    Normalize an element-set key for file paths and dictionary lookups.

    Parameters
    ----------
    key
        Raw element-set key from config or UI input.

    Returns
    -------
    str
        Lowercase normalized key.
    """
    return key.strip().lower()


def build_allowed_mask(
    structure_elements: Sequence[set[str] | Iterable[str]],
    allowed_elements: set[str] | None,
) -> np.ndarray:
    """
    Build a boolean mask for structures that pass an element filter.

    A structure is selected only when all of its elements are within
    ``allowed_elements``. If ``allowed_elements`` is ``None``, every
    structure is selected.

    Parameters
    ----------
    structure_elements
        Element symbols for each structure.
    allowed_elements
        Allowed elements. ``None`` means no filtering (all True).

    Returns
    -------
    np.ndarray
        Boolean array where ``True`` means the structure is selected.
    """
    if allowed_elements is None:
        return np.ones(len(structure_elements), dtype=bool)

    allowed = set(allowed_elements)
    return np.array(
        [set(elements).issubset(allowed) for elements in structure_elements],
        dtype=bool,
    )


def filter_sequence(values: Sequence, mask: np.ndarray) -> list:
    """
    Filter a sequence with a boolean mask and return a plain list.

    Parameters
    ----------
    values
        Sequence to filter.
    mask
        Boolean mask with the same length as ``values``.

    Returns
    -------
    list
        Filtered values as a list.
    """
    return list(np.array(values, dtype=object)[mask])


def build_element_set_masks(
    structure_elements: Sequence[set[str] | Iterable[str]],
    element_sets: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """
    Build one structure-selection mask for each configured element set.

    Parameters
    ----------
    structure_elements
        Per-structure element collections.
    element_sets
        Element-set mapping returned by ``load_element_sets``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of set key to boolean mask.
    """
    return {
        key: build_allowed_mask(structure_elements, set_info.get("elements"))
        for key, set_info in element_sets.items()
    }


def filter_results_dict(
    results: dict[str, Sequence],
    mask: np.ndarray,
) -> dict[str, list]:
    """
    Filter a results dictionary with a structure mask.

    Parameters
    ----------
    results
        Mapping of ``ref`` and model prediction arrays.
    mask
        Boolean mask selecting rows to keep.

    Returns
    -------
    dict[str, list]
        Filtered result mapping.
    """
    filtered: dict[str, list] = {}
    expected = len(mask)
    for key, values in results.items():
        if len(values) == 0:
            filtered[key] = []
            continue
        if len(values) != expected:
            raise ValueError(
                f"Length mismatch for '{key}': got {len(values)}, expected {expected}."
            )
        filtered[key] = filter_sequence(values, mask)
    return filtered


def filter_hoverdata_dict(
    hoverdata: dict[str, Sequence],
    mask: np.ndarray,
) -> dict[str, list]:
    """
    Filter hover-label columns with a structure mask.

    Parameters
    ----------
    hoverdata
        Hover column mapping used by plot decorators.
    mask
        Boolean mask selecting rows to keep.

    Returns
    -------
    dict[str, list]
        Filtered hover-label mapping.
    """
    filtered: dict[str, list] = {}
    expected = len(mask)
    for key, values in hoverdata.items():
        if len(values) != expected:
            raise ValueError(
                f"Length mismatch for hover column '{key}': got {len(values)}, "
                f"expected {expected}."
            )
        filtered[key] = filter_sequence(values, mask)
    return filtered


def write_element_sets_summary_file(
    out_path: str | Path,
    element_sets: dict[str, dict[str, Any]],
    element_set_masks: dict[str, np.ndarray],
) -> None:
    """
    Write element-set summary information to ``element_sets.json``.

    The output contains, for each set:
    1. Display name and description.
    2. Allowed elements.
    3. Number of selected structures.
    4. Original structure positions used by this set.

    Parameters
    ----------
    out_path
        Benchmark output directory under ``app/data``.
    element_sets
        Element-set mapping returned by ``load_element_sets``.
    element_set_masks
        Mapping of set key to structure-selection mask.
    """
    element_sets_data: dict[str, dict[str, Any]] = {}
    for key, set_info in element_sets.items():
        mask = element_set_masks[key]
        indices = np.flatnonzero(mask).astype(int).tolist()
        elements = set_info.get("elements")
        element_sets_data[key] = {
            "name": set_info.get("label", key),
            "description": set_info.get("description", ""),
            "elements": sorted(elements) if elements is not None else None,
            "count": len(indices),
            "indices": indices,
        }

    summary_data = {"element_sets": element_sets_data}
    output_path = Path(out_path) / "element_sets.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(summary_data, fp, indent=2)
        fp.write("\n")


def load_element_sets(config_path: str | Path) -> dict[str, dict]:
    """
    Load element-set definitions from YAML config.

    Parameters
    ----------
    config_path
        Path to ``element_sets.yml``.

    Returns
    -------
    dict[str, dict]
        Mapping keyed by normalized set key. Each value contains:
        ``label``, ``description``, and ``elements`` (``set[str]`` or ``None``).
    """
    with open(config_path) as f:
        config = safe_load(f) or {}

    raw_sets = config.get("element_sets") or {}
    element_sets: dict[str, dict] = {}

    for raw_key, set_config in raw_sets.items():
        key = normalize_element_set_key(raw_key)
        set_config = set_config or {}
        raw_elements = set_config.get("elements")
        element_sets[key] = {
            "label": set_config.get("name", raw_key),
            "description": set_config.get("description", ""),
            "elements": set(raw_elements) if raw_elements is not None else None,
        }

    if not element_sets:
        raise ValueError(f"No element sets defined in config: {config_path}")

    return element_sets
