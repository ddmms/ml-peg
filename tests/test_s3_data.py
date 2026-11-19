"""Test the S3 data helpers against the ML-PEG bucket."""

from __future__ import annotations

from ml_peg.data.data import exists


def test_exists() -> None:
    """A public file is found, and a missing one is reported as absent (403)."""
    assert exists("app/data/data.tar.gz")
    assert not exists("outputs/molecular_dynamics/binary_mixtures/no-such-model.zip")
