"""Fixtures for the browser-free Explorer tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from explorer_synthetic_data import write_synthetic_tree
import pytest

from ml_peg.analysis.explorer import build_explorer_data


@pytest.fixture(scope="session", autouse=True)
def benchmark_data(
    tmp_path_factory, clear_explorer_caches: Callable[[], None]
) -> Iterator[list[tuple[str, str]]]:
    """
    Point the Explorer at a synthetic data tree for every test in this directory.

    Keeps the tests independent of any local data under ``ml_peg/app/data``.

    Parameters
    ----------
    tmp_path_factory
        Session temporary-directory factory.
    clear_explorer_caches
        Clears the Explorer's per-process caches.

    Yields
    ------
    list[tuple[str, str]]
        The ``(category, benchmark)`` pairs in the tree.
    """
    root = tmp_path_factory.mktemp("explorer-data")
    written = write_synthetic_tree(root)
    patches = pytest.MonkeyPatch()
    patches.setattr(build_explorer_data, "DATA_ROOT", root)
    clear_explorer_caches()
    try:
        yield written
    finally:
        patches.undo()
        clear_explorer_caches()
