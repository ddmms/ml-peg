"""Test model variants in benchmark tables."""

from __future__ import annotations

import json

from dash.dash_table import DataTable

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.app.build_app import build_summary_table
from ml_peg.app.utils.utils import filter_rows_by_models
from ml_peg.models.get_models import get_model_names


def test_build_table_model_variants(tmp_path):
    """Keep model variants distinct while associating them with one base model."""
    base_model = get_model_names()[0]
    corrected_model = f"{base_model}-test-D3"
    table_path = tmp_path / "table.json"

    @build_table(
        filename=table_path,
        thresholds={
            "Force MAE": {
                "good": 0.0,
                "bad": 1.0,
                "unit": "eV/A",
            }
        },
        model_variants={corrected_model: base_model},
        summary_model_ids={base_model: corrected_model},
    )
    def make_table():
        return {
            "Force MAE": {
                base_model: 0.2,
                corrected_model: 0.1,
            }
        }

    make_table()

    with table_path.open(encoding="utf8") as file:
        table = json.load(file)

    rows = {row["id"]: row for row in table["data"]}
    assert rows[base_model]["Force MAE"] == 0.2
    assert rows[corrected_model]["Force MAE"] == 0.1
    assert table["model_name_map"][corrected_model] == base_model
    assert table["summary_model_ids"][base_model] == corrected_model

    filtered_rows = filter_rows_by_models(
        table["data"], [base_model], table["model_name_map"]
    )
    assert {row["id"] for row in filtered_rows} == {base_model, corrected_model}


def test_summary_table_uses_preferred_variant():
    """Use the configured corrected variant for aggregate benchmark scores."""
    base_model = get_model_names()[0]
    corrected_model = f"{base_model}-test-D3"
    benchmark_table = DataTable(
        data=[
            {"MLIP": base_model, "id": base_model, "Score": 0.2},
            {"MLIP": corrected_model, "id": corrected_model, "Score": 0.8},
        ],
        columns=[
            {"name": "MLIP", "id": "MLIP"},
            {"name": "Score", "id": "Score"},
        ],
    )
    benchmark_table.description = "Test benchmark"
    benchmark_table.model_name_map = {
        base_model: base_model,
        corrected_model: base_model,
    }
    benchmark_table.summary_model_ids = {base_model: corrected_model}

    summary = build_summary_table({"Cluster Forces": benchmark_table})
    model_row = next(row for row in summary.data if row["MLIP"] == base_model)

    assert model_row["Cluster Forces Score"] == 0.8
