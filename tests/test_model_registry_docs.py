"""Tests for the directive documenting the model registry."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from yaml import safe_load

sys.path.append(str(Path(__file__).parents[1] / "docs" / "source" / "_ext"))

from model_registry import (  # noqa: E402
    MODELS_FILE,
    NOTES_FILE,
    build_rst,
    model_families,
    registry_entries,
)

ENTRY_WITH_BLANK_LINE = """\
spaced-model:
  module: ase.calculators.emt
  class_name: EMT

  trained_on_dispersion: true
  kwargs:
    asap_cutoff: false

next-model:
  module: ase.calculators.emt
  class_name: EMT
  trained_on_dispersion: true
"""


def test_entries_keep_fields_after_blank_lines(tmp_path: Path):
    """A blank line within an entry does not truncate its documentation."""
    models_file = tmp_path / "models.yml"
    models_file.write_text(ENTRY_WITH_BLANK_LINE, encoding="utf8")

    entries = registry_entries(models_file)

    assert sorted(entries) == ["next-model", "spaced-model"]
    assert "trained_on_dispersion: true" in entries["spaced-model"]
    assert "asap_cutoff: false" in entries["spaced-model"]
    assert "next-model" not in entries["spaced-model"]


def test_every_model_is_documented():
    """Every model in the registry is documented under a family."""
    registry = safe_load(MODELS_FILE.read_text(encoding="utf8"))
    notes = safe_load(NOTES_FILE.read_text(encoding="utf8"))

    families = model_families(registry_entries(), notes)
    documented = [name for names in families.values() for name in names]

    assert sorted(documented) == sorted(registry)


def test_documented_configuration_matches_registry():
    """Each documented entry is the configuration the model is loaded with."""
    registry = safe_load(MODELS_FILE.read_text(encoding="utf8"))
    notes = safe_load(NOTES_FILE.read_text(encoding="utf8"))
    entries = registry_entries()
    rst = build_rst(entries, notes)

    for name, entry in entries.items():
        assert safe_load(entry) == {name: registry[name]}
        assert f"   {entry.splitlines()[0]}" in rst


def test_family_is_required_for_every_class_name():
    """A model family that is not described fails the docs build."""
    entries = registry_entries()
    notes = safe_load(NOTES_FILE.read_text(encoding="utf8"))
    notes["families"].pop("MACE")

    with pytest.raises(ValueError, match="belongs to no family"):
        model_families(entries, notes)
