"""Check saved plots remain usable with legacy Plotly template defaults."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ml_peg.app.bulk_crystal.equation_of_state_acwf import (
    app_equation_of_state_acwf as acwf,
)


def test_acwf_registers_box_plot_with_legacy_template(tmp_path, monkeypatch):
    """Load old template defaults without reconstructing a validated Figure."""
    figure = {
        "data": [{"type": "box", "y": [1, 2, 3]}],
        "layout": {
            "template": {"data": {"scattermapbox": [{"type": "scattermapbox"}]}}
        },
    }
    path = tmp_path / "relative_errors_box.json"
    path.write_text(json.dumps(figure))
    monkeypatch.setattr(acwf, "DATA_PATH", tmp_path)
    monkeypatch.setattr(acwf, "BOX_PLOT_PATH", path)
    monkeypatch.setattr(acwf, "MODELS", ["example"])
    monkeypatch.setattr(acwf, "callback", lambda *a, **kw: lambda f: f)
    captured = {}
    monkeypatch.setattr(acwf, "plot_from_table_cell", lambda **kw: captured.update(kw))

    acwf.EquationOfStateACWFApp.register_callbacks(SimpleNamespace(table_id="test"))

    plots = captured["cell_to_plot"]["example"]
    for column in acwf.BOX_PLOT_COLUMNS:
        assert plots[column].figure == figure
