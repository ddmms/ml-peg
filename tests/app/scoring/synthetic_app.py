"""Small deterministic dataset served through the production app and callbacks.

Run in a separate process so Dash's callback registry and imported benchmark
modules cannot leak between this fixture and the real-data browser tests.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from dash import Dash
from werkzeug.serving import make_server

from ml_peg.app import APP_ROOT, build_app
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils import build_components
from ml_peg.app.utils.head_scripts import inject_head_scripts

MODELS = ["mace-mp-0a", "mace-mp-0b3", "mace-mpa-0"]


class SyntheticBenchmark(BaseApp):
    """Use the normal table, controls, filters, and stores without model runs."""

    def register_callbacks(self):
        """Scoring callbacks are registered by the production layout builder."""


def serve(root: Path):
    """Build the fixed fixture under root and publish its ephemeral port."""
    assets = root / "assets"
    shutil.copytree(APP_ROOT / "data" / "ui", assets / "ui")
    app = Dash(
        __name__,
        assets_folder=str(assets),
        title="ML-PEG",
        suppress_callback_exceptions=True,
    )
    inject_head_scripts(app)
    build_app.MODELS = MODELS
    build_components.MODELS = MODELS

    # Explicit baseline scores are hand calculated, not produced by the scorer.
    specs = [
        (
            "Alpha",
            "B1",
            ["mace-polar-1", "mace-multihead"],
            ["H"],
            [(2, 8, 0.5), (6, 4, 0.5), (2, None, 0.8)],
        ),
        (
            "Alpha",
            "B2",
            ["mace-multihead"],
            ["O"],
            [(7, 7, 0.3), (5, 5, 0.5), ("NaN", 7, "NaN")],
        ),
        ("Beta", "B3", ["mlip_audit"], ["C"], [(1, 1, 0.9), (3, 3, 0.7), (1, 1, 0.9)]),
    ]
    apps, layouts, tables, frameworks = {}, {}, {}, {}
    for category, name, tags, elements, values in specs:
        folder = assets / category / name
        folder.mkdir(parents=True)
        payload = {
            "data": [
                {"MLIP": model, "id": model, "M1": x, "M2": y, "Score": score}
                for model, (x, y, score) in zip(MODELS, values, strict=True)
            ],
            "columns": [
                {"id": key, "name": key} for key in ("MLIP", "Score", "M1", "M2")
            ],
            "tooltip_header": {key: key for key in ("MLIP", "Score", "M1", "M2")},
            "thresholds": {
                key: {"good": 0, "bad": 10, "unit": "eV"} for key in ("M1", "M2")
            },
            "weights": {"M1": 1, "M2": 1},
        }
        (folder / "table.json").write_text(json.dumps(payload))
        (folder / "info.json").write_text(json.dumps({"elements": elements}))
        benchmark = SyntheticBenchmark(
            name=name,
            description="Synthetic scoring fixture",
            table_path=folder / "table.json",
            extra_components=[],
            info_path=folder / "info.json",
            framework_ids=tags,
            include_ml_peg=False,
        )
        apps[name] = benchmark
        layouts.setdefault(category, {})[name] = benchmark.layout
        tables.setdefault(category, {})[name] = benchmark.table
        frameworks.setdefault(category, {})[name] = tags

    build_app.get_all_tests = lambda **kwargs: (apps, layouts, tables, frameworks)
    build_app.build_full_app(app)
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    (root / "url").write_text(f"http://127.0.0.1:{server.server_port}")
    server.serve_forever()


if __name__ == "__main__":
    serve(Path(sys.argv[-1]))
