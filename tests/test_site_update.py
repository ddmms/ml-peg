"""The footer reports recorded updates, rather than visits or server restarts."""

from __future__ import annotations

from datetime import datetime, timezone
import os

from ml_peg.app.utils import build_components
from ml_peg.app.utils.site_update import site_updated_at


def test_update_uses_latest_build_or_results(tmp_path):
    """Either a code build or a later result refresh can advance the timestamp."""
    stamp = tmp_path / "_build_time.txt"
    stamp.write_text("2026-11-12T10:30:00Z\n")
    built = datetime(2026, 11, 12, 10, 30, tzinfo=timezone.utc)
    result = tmp_path / "data/category/benchmark/table.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}")
    os.utime(result, (built.timestamp() - 3600,) * 2)
    assert site_updated_at(tmp_path) == built
    refreshed = built.timestamp() + 3600
    os.utime(result, (refreshed,) * 2)
    assert site_updated_at(tmp_path).timestamp() == refreshed
    # Rebuilding the layout does not substitute the current time.
    assert site_updated_at(tmp_path).timestamp() == refreshed


def test_missing_metadata_and_ui_files_do_not_invent_update(tmp_path):
    """An empty install or changed UI auxiliary file is not a result refresh."""
    assert site_updated_at(tmp_path) is None
    ui_file = tmp_path / "data/ui/example/settings.json"
    ui_file.parent.mkdir(parents=True)
    ui_file.write_text("{}")
    assert site_updated_at(tmp_path) is None


def test_footer_shows_utc_and_machine_readable_time(monkeypatch):
    """The footer has a clear UTC label and an accessible semantic timestamp."""
    updated = datetime(2026, 11, 12, 10, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(build_components, "site_updated_at", lambda: updated)
    monkeypatch.setattr(build_components, "__version__", "1.2.3")
    footer = build_components.build_footer()
    info = next(
        c for c in footer.children if getattr(c, "id", None) == "site-build-info"
    )
    assert info.children[0].children == "ML-PEG v1.2.3"
    label = next(
        c for c in info.children if getattr(c, "id", None) == "site-last-updated"
    )
    assert label.children[0] == "Last updated: "
    assert label.children[1].children == "12 Nov 2026, 10:30 UTC"
    assert label.children[1].dateTime == updated.isoformat()
    monkeypatch.setattr(build_components, "site_updated_at", lambda: None)
    info = next(
        c
        for c in build_components.build_footer().children
        if getattr(c, "id", None) == "site-build-info"
    )
    assert info.children[0].children == "ML-PEG v1.2.3"
    assert not any(getattr(c, "id", None) == "site-last-updated" for c in info.children)
