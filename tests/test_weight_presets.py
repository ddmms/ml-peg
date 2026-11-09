from __future__ import annotations

from yaml import safe_load

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.weight_presets import WEIGHT_PRESETS


def test_preset_categories_match_category_titles():
    category_files = [
        path for path in APP_ROOT.glob("*/*.yml") if path.parent.name != "utils"
    ]
    category_titles = {safe_load(path.read_text())["title"] for path in category_files}

    unknown = {
        preset: [title for title in titles if title not in category_titles]
        for preset, titles in WEIGHT_PRESETS.items()
    }
    unknown = {preset: titles for preset, titles in unknown.items() if titles}

    assert not unknown
