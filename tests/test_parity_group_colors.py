"""Check optional parity group colours and unchanged default styling."""

from __future__ import annotations

import json

import pytest

from ml_peg.analysis.utils.decorators import plot_parity


@pytest.mark.parametrize("colored", [False, True])
def test_parity_group_colors(tmp_path, colored):
    """
    Match point and legend colours without changing point identifiers.

    Parameters
    ----------
    tmp_path
        Temporary output directory.
    colored
        Whether to override model colours with group colours.
    """
    path = tmp_path / "parity.json"
    colors = {"yes": "green", "no": "orange"} if colored else None
    result = {"model": [1, 2, 3], "ref": [1.1, 1.9, 3.2]}
    plot_parity(
        filename=path,
        symbol_by=["yes", "no", "yes"],
        symbol_labels={"yes": "Converged", "no": "Unconverged"},
        group_colors=colors,
        hoverdata={"Reaction": ["3", "8", "10"]},
        log=True,
    )(lambda: result)()
    figure = json.loads(path.read_text())
    points, yes, no = figure["data"][:3]
    assert points["customdata"] == [["3"], ["8"], ["10"]]
    assert points["marker"]["symbol"] == ["circle", "square", "circle"]
    assert yes["name"] == "Converged"
    assert no["name"] == "Unconverged"
    if colored:
        assert points["marker"]["color"] == ["green", "orange", "green"]
        assert yes["marker"]["color"] == "green"
        assert no["marker"]["color"] == "orange"
    else:
        assert "color" not in points["marker"]
        assert yes["marker"]["color"] == no["marker"]["color"] == "black"


def test_missing_group_color(tmp_path):
    """
    Reject incomplete mappings rather than silently mislabelling groups.

    Parameters
    ----------
    tmp_path
        Temporary output directory.
    """
    with pytest.raises(ValueError, match="Missing group colours.*no"):
        plot_parity(
            filename=tmp_path / "parity.json",
            symbol_by=["yes", "no"],
            group_colors={"yes": "green"},
            log=True,
        )(lambda: {"model": [1, 2], "ref": [1, 2]})()
