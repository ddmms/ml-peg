"""Pytest CLI options for the polymers benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

POLYMER_SET_DIR = Path(__file__).parent / "resources" / "polymer_sets"


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register the polymer-benchmark CLI options.

    Note that ``--models`` (used to restrict to a subset of MLIPs from
    ``models.yml``) is registered by the root ``conftest.py``, not here.

    Parameters
    ----------
    parser
        Pytest argument parser injected by the framework.
    """
    parser.addoption("--poly-id", action="store", default=None, type=str)
    parser.addoption("--poly-index", action="store", default=None, type=int)
    parser.addoption("--poly-set", action="store", default="small", type=str)
    parser.addoption("--time-prefactor", action="store", default=1.0, type=float)


def _poly_id_from_index(polymer_set: str, poly_index: int) -> str:
    """
    Return a polymer id from a named set using a one-based index.

    Parameters
    ----------
    polymer_set
        Name of the polymer set file, without ``.txt``.
    poly_index
        One-based index into the polymer set file.

    Returns
    -------
    str
        Polymer id at ``poly_index`` in the selected set file.
    """
    if poly_index < 1:
        raise pytest.UsageError("--poly-index is one-based and must be at least 1.")

    set_path = POLYMER_SET_DIR / f"{polymer_set}.txt"
    if not set_path.exists():
        raise pytest.UsageError(f"Unknown --poly-set '{polymer_set}': {set_path}")

    polymer_ids = [
        line.strip() for line in set_path.read_text(encoding="utf-8").splitlines()
    ]
    polymer_ids = [poly_id for poly_id in polymer_ids if poly_id]
    if poly_index > len(polymer_ids):
        raise pytest.UsageError(
            f"--poly-index {poly_index} is out of range for --poly-set "
            f"'{polymer_set}' ({len(polymer_ids)} polymers)."
        )
    return polymer_ids[poly_index - 1]


@pytest.fixture
def poly_id(request: pytest.FixtureRequest) -> str:
    """
    Return the ``--poly-id`` value (a polymer identifier from data.csv).

    Parameters
    ----------
    request
        Pytest fixture request injected by the framework.

    Returns
    -------
    str
        The value passed to ``--poly-id`` on the command line, or the polymer id
        selected by ``--poly-set`` and ``--poly-index``.
    """
    cli_poly_id = request.config.getoption("--poly-id")
    if cli_poly_id is not None:
        return str(cli_poly_id)

    poly_index = request.config.getoption("--poly-index")
    if poly_index is None:
        raise pytest.UsageError("Pass either --poly-id or --poly-index.")

    return _poly_id_from_index(
        polymer_set=str(request.config.getoption("--poly-set")),
        poly_index=int(poly_index),
    )


@pytest.fixture
def time_prefactor(request: pytest.FixtureRequest) -> float:
    """
    Return the ``--time-prefactor`` value (multiplier on all stage durations).

    Parameters
    ----------
    request
        Pytest fixture request injected by the framework.

    Returns
    -------
    float
        The value passed to ``--time-prefactor`` on the command line.
    """
    return float(request.config.getoption("--time-prefactor"))
