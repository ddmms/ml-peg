"""Verify plot CSV downloads using Plotly rendered in Chromium."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ASSET = (
    Path(__file__).resolve().parents[2]
    / "ml_peg/app/data/ui/table_download/plot_capture.js"
)


def test_csv_exports_rendered_numpy_hover_data(page):
    """Exercise Plotly's actual binary-array decoding in the browser."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    figure = go.Figure(
        go.Scatter(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            customdata=np.array([[10.0], [20.0]]),
            hovertemplate="<b>Energy: </b>%{customdata[0]:.3f}<extra></extra>",
        )
    )
    page.set_content('<div id="test"></div>')
    page.add_script_tag(content=get_plotlyjs())
    page.evaluate(
        '(fig) => Plotly.newPlot("test", fig.data, fig.layout)',
        json.loads(figure.to_json()),
    )
    page.add_script_tag(path=str(ASSET))
    result = page.evaluate(
        '() => window.dash_clientside.plot_download.downloadPlot(1, "csv", "test")'
    )
    rows = list(csv.DictReader(io.StringIO(result["content"])))
    assert [(row["x"], row["y"], row["Energy"]) for row in rows] == [
        ("1", "3", "10"),
        ("2", "4", "20"),
    ]
