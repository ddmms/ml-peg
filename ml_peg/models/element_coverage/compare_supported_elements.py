"""
Finds overlap of supported elements with common datasets.

Reads the probe output (``model_supported_elements.json`` from
``find_supported_elements.py``) and the per-dataset element lists in
``element_coverage.json``. For every model it shows which datasets its supported
elements overlap with, and the elements it covers beyond them. This is a starting
point, not a recommendation: you still have to check the overlap matches the
model's actual training before setting ``datasets`` / ``additional_supported_elements``.

Note: the probe measures "does the model error on this element", which equals real
coverage only for models that validate elements (e.g. MACE). Permissive models
(e.g. orb accepts any element) over-report and overlap with every dataset.

Run with::

    python ml_peg/models/element_coverage/compare_supported_elements.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ml_peg.analysis.utils.periodic_table import PERIODIC_TABLE_SYMBOLS
from ml_peg.app import APP_ROOT

EMPIRICAL_PATH = Path(__file__).parent / "model_supported_elements.json"
COVERAGE_PATH = APP_ROOT / "data" / "element_coverage.json"

_ORDER = {symbol: i for i, symbol in enumerate(PERIODIC_TABLE_SYMBOLS)}


def _sorted(symbols: set[str]) -> list[str]:
    """
    Return symbols in periodic-table order.

    Parameters
    ----------
    symbols
        Element symbols to order.

    Returns
    -------
    list[str]
        Symbols sorted by atomic number.
    """
    return sorted(symbols, key=lambda s: _ORDER.get(s, len(_ORDER)))


def main() -> None:
    """
    Suggest ``datasets`` tags for each model from its empirical coverage.

    A dataset "matches" a model when the model supports every element that
    dataset covers (its coverage is a subset of the model's empirical set). For
    each model the matching datasets are listed largest-first, and a suggested
    ``datasets`` tag (their union) plus the ``additional_supported_elements``
    beyond it are printed.
    """
    empirical = json.loads(EMPIRICAL_PATH.read_text())
    coverage = json.loads(COVERAGE_PATH.read_text())["datasets"]
    cov_sets = {
        name: set(data["supported"])
        for name, data in coverage.items()
        if data.get("supported")
    }

    for model in sorted(empirical):
        emp = set(empirical[model])
        matches = sorted(
            ((name, s) for name, s in cov_sets.items() if s <= emp),
            key=lambda item: -len(item[1]),
        )

        print(f"\n{model}  (empirical: {len(emp)})")
        if not matches:
            print("  no dataset fully covered - tag with additional only:")
            print(f"    additional_supported_elements: [{', '.join(_sorted(emp))}]")
            continue

        print("  overlapping datasets (coverage ⊆ model):")
        for name, s in matches:
            extra = "exact" if s == emp else f"+{len(emp - s)} more"
            print(f"    {name} (covers {len(s)}) {extra}")

        # Tag the model with one or more of the matches above (whichever reflect
        # its training provenance; datasets combine as a union). additional is
        # computed against the largest match, so it holds only for a tag that
        # includes that dataset.
        base = matches[0][1]
        additional = _sorted(emp - base)
        elems = ", ".join(additional) if additional else "none"
        print(f"  -> tag with 1+ above; additional_supported_elements: [{elems}]")


if __name__ == "__main__":
    main()
