"""Utility functions for filtering data."""

from __future__ import annotations

from collections.abc import Callable

from dash.dcc import Graph
import numpy as np


def filter_parity(
    filter_elements: set[str],
    data: Graph,
    test_elements: list[set[str]],
    metric_getter: Callable,
    mask_to_getter: bool = False,
) -> dict[str, dict]:
    """
    Apply elements filter to data.

    Parameters
    ----------
    filter_elements
        Set of elements to filter out of data.
    data
        Scatter plot to filter.
    test_elements
        List of element for each system.
    metric_getter
        Function to calculate metric from filtered data.
    mask_to_getter
        Whether to pass the mask to the metric getter.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    # Get overlap of deselected elements with each system's elements
    filtered_indices = [
        not bool(elements & filter_elements) for elements in test_elements
    ]

    results = {}
    ref_filtered = False

    for plot in data.figure.data:
        # Ignore unamed (parity) line
        if plot.name:
            print()
            print("PLOT NAME", plot.name)
            results[plot.name] = np.array(plot.x)[filtered_indices].tolist()
            print("RESULTS", results[plot.name])
            print()
            if not ref_filtered:
                results["ref"] = np.array(plot.y)[filtered_indices].tolist()
                ref_filtered = True

    if mask_to_getter:
        return metric_getter(results, mask=filtered_indices)
    return metric_getter(results)
