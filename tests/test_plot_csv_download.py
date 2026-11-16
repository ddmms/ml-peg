"""Exercise the browser CSV callback with real JavaScript execution."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ASSET = (
    Path(__file__).resolve().parents[1]
    / "ml_peg/app/data/ui/table_download/plot_capture.js"
)


def export_csv(traces: str) -> list[dict[str, str]]:
    """Run the callback against a minimal rendered-graph stub."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for clientside callback tests")
    script = """
const fs = require('node:fs');
const vm = require('node:vm');
global.window = {dash_clientside: {no_update: null}};
const plot = {data: TRACES, classList: {contains: () => true}};
global.document = {getElementById: () => plot};
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));
const result = window.dash_clientside.plot_download.downloadPlot(1, 'csv', 'test');
process.stdout.write(JSON.stringify(result));
""".replace("TRACES", traces)
    result = subprocess.run(
        [node, "-e", script, str(ASSET)],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(json.loads(result.stdout)["content"])))


def test_csv_includes_hover_data_across_traces():
    """Preserve precision, escaped text, and row alignment across mixed traces."""
    rows = export_csv(
        json.dumps(
            [
                {
                    "name": "model",
                    "x": [1, 2],
                    "y": [3, 4],
                    "text": ['system, "A"\nsecond line', "B"],
                    "hovertext": "shared hover",
                    "customdata": [["A", 0.123456789], ["B"]],
                },
                {"name": "other", "x": [5], "y": [6], "customdata": [42]},
                {"name": "__parity_line__", "x": [0], "y": [0]},
            ]
        )
    )
    assert rows[0] == {
        "trace": "model",
        "point_index": "0",
        "x": "1",
        "y": "3",
        "text": 'system, "A"\nsecond line',
        "hovertext": "shared hover",
        "customdata[0]": "A",
        "customdata[1]": "0.123456789",
    }
    assert rows[1]["customdata[1]"] == ""
    assert rows[1]["text"] == "B"
    assert rows[2]["customdata[0]"] == "42"
    assert rows[2]["hovertext"] == ""
    assert rows[3]["trace"] == "y=x"
    assert rows[3]["customdata[0]"] == ""


def test_csv_without_hover_data_keeps_original_columns():
    """Simple plots retain their schema and support typed coordinate arrays."""
    assert export_csv("[{x: new Float64Array([1, 2]), y: [3]}]") == [
        {"trace": "trace_0", "point_index": "0", "x": "1", "y": "3"},
        {"trace": "trace_0", "point_index": "1", "x": "2", "y": ""},
    ]


def test_csv_uses_hover_labels_and_aligns_different_trace_schemas():
    """The same index can have different labels across traces, and vice versa."""
    rows = export_csv(
        json.dumps(
            [
                {
                    "x": [1],
                    "y": [2],
                    "customdata": [["water", 0.123456789]],
                    "hovertemplate": (
                        "<b>System: </b>%{customdata[0]}<br>"
                        "Error: %{customdata[1]:.3f}<extra></extra>"
                    ),
                },
                {
                    "x": [3],
                    "y": [4],
                    "customdata": [[0.5, "methane", 9]],
                    "hovertemplate": (
                        "Error=%{customdata[0]}<br>"
                        "<b>System</b>: %{customdata[1]}<br>"
                        "Count: %{customdata[2]}"
                    ),
                },
            ]
        )
    )
    assert rows[0]["System"] == "water"
    assert rows[0]["Error"] == "0.123456789"
    assert rows[0]["Count"] == ""
    assert rows[1]["System"] == "methane"
    assert rows[1]["Error"] == "0.5"
    assert rows[1]["Count"] == "9"
    assert not any(key.startswith("customdata") for key in rows[0])


def test_csv_handles_duplicate_reserved_and_unlabelled_fields():
    """Keep all values when labels collide or a template has no field label."""
    rows = export_csv(
        json.dumps(
            [
                {
                    "x": [1],
                    "y": [2],
                    "customdata": [[3, 4, 5, 6]],
                    "hovertemplate": (
                        "x: %{customdata[0]}<br>Error: %{customdata[1]}<br>"
                        "Error: %{customdata[2]}<br>%{customdata[3]}"
                    ),
                },
            ]
        )
    )
    assert rows[0]["x"] == "1"
    assert rows[0]["x (2)"] == "3"
    assert rows[0]["Error"] == "4"
    assert rows[0]["Error (2)"] == "5"
    assert rows[0]["customdata[3]"] == "6"
