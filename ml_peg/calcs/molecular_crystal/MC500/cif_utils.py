"""CIF handling for the MC500 molecular-crystal benchmark."""

from __future__ import annotations

from os import PathLike

from ase import Atoms
from ase.io import read
import numpy as np

BOUNDARY_TOLERANCE = 1e-12
ORIGIN_SHIFT = 1e-8
CIF_LABEL_ARRAY = "cif_labels"


def _validate_cif(atoms: Atoms, filename: str | PathLike[str]) -> None:
    """
    Validate assumptions made by the MC500 relaxation protocol.

    Parameters
    ----------
    atoms
        Structure parsed from CIF.
    filename
        Source CIF path, used in error messages.

    Raises
    ------
    ValueError
        If the structure has no full-rank periodic cell, has partial occupancies,
        or does not contain enough label information to track expanded atoms.
    """
    if atoms.cell.rank != 3 or not atoms.pbc.all():
        raise ValueError(f"MC500 CIF must be fully periodic: {filename}")

    occupancies = atoms.info.get("_atom_site_occupancy")
    if occupancies is None or not np.allclose(
        np.asarray(occupancies, dtype=float), 1.0
    ):
        raise ValueError(
            f"MC500 CIF contains missing or partial occupancies: {filename}"
        )

    labels = atoms.info.get("_atom_site_label")
    kinds = atoms.arrays.get("spacegroup_kinds")
    if labels is None or kinds is None or (len(kinds) and max(kinds) >= len(labels)):
        raise ValueError(f"MC500 CIF has inconsistent atom labels: {filename}")


def _store_expanded_labels(atoms: Atoms) -> None:
    """
    Attach each expanded atom's original asymmetric-unit label.

    Parameters
    ----------
    atoms
        Structure containing CIF labels and ASE ``spacegroup_kinds``.
    """
    labels = np.asarray(atoms.info["_atom_site_label"], dtype=str)
    expanded_labels = labels[atoms.arrays["spacegroup_kinds"]]
    atoms.new_array(CIF_LABEL_ARRAY, expanded_labels)


def _move_from_cell_boundaries(atoms: Atoms) -> None:
    """
    Apply a common origin shift when atoms lie exactly on a cell boundary.

    The translation preserves all periodic geometry while avoiding ambiguity between
    fractional coordinates zero and one in later CIF or trajectory serialization.

    Parameters
    ----------
    atoms
        Structure whose fractional coordinates will be normalized in place.
    """
    scaled = atoms.get_scaled_positions(wrap=True)
    on_boundary = np.isclose(
        scaled, 0.0, atol=BOUNDARY_TOLERANCE, rtol=0.0
    ) | np.isclose(scaled, 1.0, atol=BOUNDARY_TOLERANCE, rtol=0.0)
    if on_boundary.any():
        atoms.set_scaled_positions(np.mod(scaled + ORIGIN_SHIFT, 1.0))
    else:
        atoms.set_scaled_positions(scaled)


def read_mc500_cif(filename: str | PathLike[str]) -> Atoms:
    """
    Read and normalize an MC500 reference CIF for relaxation.

    ASE expands the CIF symmetry operations into a complete conventional unit cell.
    The resulting ``Atoms`` object is therefore an explicit P1 representation for the
    calculator, while the original space-group metadata and asymmetric-unit labels are
    retained for validation and traceability.

    Parameters
    ----------
    filename
        Path to an MC500 CIF file.

    Returns
    -------
    Atoms
        Full-cell periodic structure with wrapped fractional coordinates and an
        expanded ``cif_labels`` array.
    """
    atoms = read(
        filename,
        format="cif",
        reader="pycodcif",
        store_tags=True,
        primitive_cell=False,
        subtrans_included=True,
        fractional_occupancies=True,
    )
    _validate_cif(atoms, filename)
    _store_expanded_labels(atoms)
    _move_from_cell_boundaries(atoms)
    return atoms
