"""Preview caching preserves selection, invalidation and missing-data behaviour."""

from __future__ import annotations

from dash.exceptions import PreventUpdate
import pytest

from ml_peg.app.physicality.diatomics import curve_utils as curves


@pytest.fixture
def preview(tmp_path, monkeypatch):
    """Count renders with source-dependent output, avoiding Matplotlib allocations."""
    curves._cached_preview_png.cache_clear()
    calls = []

    def render(*, curve_dir, model_name, element_value, **kwargs):
        files = curves._selected_curve_files(curve_dir, model_name, element_value)
        if not files:
            raise PreventUpdate
        calls.append((model_name, element_value))
        return b"".join(p.read_bytes() for p in files), 6000, 3000

    monkeypatch.setattr(curves, "render_periodic_curve_gallery_png", render)
    for model in ("A", "B"):
        folder = tmp_path / model
        folder.mkdir()
        for pair in ("H-H", "H-O", "O-O"):
            (folder / f"{pair}.json").write_text(f"{model}:{pair}")
    yield tmp_path, calls
    curves._cached_preview_png.cache_clear()


def get(root, model="A", element="All"):
    """Request a normal screen preview."""
    return curves._preview_png(root, model, element, "All")


def test_repeat_selection_and_model_separation(preview):
    """Repeated selections reuse bytes without mixing models or element views."""
    root, calls = preview
    overview = get(root)
    assert get(root) == overview
    assert get(root, element=None) == overview
    assert get(root, "B") != overview
    assert get(root, element="H") != overview
    assert len(calls) == 3


@pytest.mark.parametrize("change", ["edit", "add", "remove", "replace"])
def test_source_changes_invalidate(preview, change):
    """Changes to the selected dataset cannot return the previous cached image."""
    root, calls = preview
    old = get(root)
    path = root / "A/H-H.json"
    if change == "edit":
        path.write_text("new curve")
    elif change == "add":
        (root / "A/C-C.json").write_text("new curve")
    elif change == "remove":
        path.unlink()
    else:
        replacement = root / "replacement.json"
        replacement.write_text("new curve")
        replacement.replace(path)
    assert get(root) != old
    assert len(calls) == 2


def test_unrelated_curve_does_not_invalidate(preview):
    """Editing a heteronuclear curve leaves the homonuclear preview cached."""
    root, calls = preview
    old = get(root)
    (root / "A/H-O.json").write_text("new heteronuclear curve")
    assert get(root) == old
    assert len(calls) == 1


def test_missing_data_recovers(preview):
    """An empty selection is not cached and becomes available when data arrives."""
    root, calls = preview
    with pytest.raises(PreventUpdate):
        get(root, element="C")
    (root / "A/C-C.json").write_text("new curve")
    assert get(root, element="C")[0] == b"new curve"
    assert len(calls) == 1


def test_cache_is_bounded(preview):
    """Visiting many model previews evicts older entries."""
    root, calls = preview
    for number in range(18):
        folder = root / str(number)
        folder.mkdir()
        (folder / "H-H.json").write_text(str(number))
        get(root, str(number))
    assert curves._cached_preview_png.cache_info().currsize == 16
    get(root, "0")
    assert len(calls) == 19


def test_file_removed_before_stat(preview, monkeypatch):
    """A disappearing source still allows the remaining curves to render."""
    root, calls = preview
    select = curves._selected_curve_files
    missing = root / "A/H-H.json"

    def remove_after_selection(*args):
        paths = select(*args)
        missing.unlink(missing_ok=True)
        return paths

    monkeypatch.setattr(curves, "_selected_curve_files", remove_after_selection)
    assert get(root)[0] == b"A:O-O"
    assert len(calls) == 1
