"""Focused regression tests for shared calculation utilities."""

from __future__ import annotations

from pathlib import Path

from ml_peg.calcs.utils.utils import extract_zip


def test_extract_zip_returns_parent_for_non_zip():
    """A raw file needs no extraction or disk access to resolve its directory."""
    filename = Path(__file__).with_name("example.extxyz")
    assert extract_zip(filename) == filename.parent
