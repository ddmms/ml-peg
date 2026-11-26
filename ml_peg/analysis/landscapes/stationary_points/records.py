"""
Read one model's stationary point outputs by structure id.

``read_by_id`` reads a model's records of one kind, and ``complete_records``
returns them in the reference's order only if they are exactly the reference's
structures: each present once, and no others. Otherwise each returns None, with
a warning, so that a model's incomplete outputs are never scored on the subset
that happens to exist, and a gap cannot shift one structure's value onto another.
"""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase import Atoms
from ase.io import read


def structure_id(molecule: str, label: str) -> str:
    """
    Get a structure's id: its molecule and label, joined.

    The label alone is not unique, since ``min00`` exists once per molecule.

    Parameters
    ----------
    molecule
        Molecule key.
    label
        Structure label within the molecule.

    Returns
    -------
    str
        The id.
    """
    return f"{molecule}_{label}"


def read_by_id(directory: Path, pattern: str) -> dict[str, Atoms] | None:
    """
    Read every record in one model's files of one kind, by structure id.

    Parameters
    ----------
    directory
        The model's output directory.
    pattern
        Glob matching the files.

    Returns
    -------
    dict[str, ase.atoms.Atoms] | None
        The records in file order, empty if no file matches; None, with a
        warning, if a file cannot be read or a structure appears twice.
    """
    records: dict[str, Atoms] = {}
    for path in sorted(directory.glob(pattern)):
        try:
            for atoms in read(path, index=":"):
                key = structure_id(atoms.info["molecule"], atoms.info["label"])
                if key in records:
                    raise ValueError(f"{key} appears twice")
                records[key] = atoms
        except Exception as exc:
            warn(f"Could not read {path}: {exc}", stacklevel=2)
            return None
    return records


def complete_records(
    found: dict[str, Atoms] | None, expected: list[str], description: str
) -> tuple[Atoms, ...] | None:
    """
    Put one model's records in the reference's order, if they match it exactly.

    Parameters
    ----------
    found
        The model's records by id, or None if they could not be read.
    expected
        Every reference structure's id, in the order to return them.
    description
        The model and the kind of output, for the warning.

    Returns
    -------
    tuple[ase.atoms.Atoms, ...] | None
        One record per expected id; None, with a warning, if any is missing or
        a record is not in the reference, which would mean the reference
        itself is incomplete.
    """
    if found is None:
        return None
    reference = set(expected)
    missing = [key for key in expected if key not in found]
    unexpected = [key for key in found if key not in reference]
    if missing or unexpected:
        warn(
            f"{description}: {len(missing)} of {len(expected)} structures missing "
            f"and {len(unexpected)} not in the reference, such as "
            f"{(missing or unexpected)[0]}; the metrics built on them are NaN",
            stacklevel=2,
        )
        return None
    return tuple(found[key] for key in expected)
