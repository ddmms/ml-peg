"""
Ti64 phonons CASTEP phonon dispersion reader.

This module provides a lightweight parser for CASTEP phonon output files to
extract phonon frequencies (in THz) and q-point coordinates. Optionally, the
q-path can be split into high-symmetry segments matching a provided k-path.

Notes
-----
- This parser assumes the CASTEP output uses THz units for phonon frequencies.
- Segment detection assumes ``kpath_in`` matches the k-path used in the CASTEP
  run.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import ase.io
import numpy as np


class PhononFromCastep:
    """
    Extract phonon frequencies and k-points from a CASTEP phonon calculation.

    Parameters
    ----------
    castep_file
        Path to a CASTEP output file readable by ASE.
    kpath_in
        Optional k-path (high-symmetry points) used to split the parsed
        q-points into per-segment index lists (``kpath_idx``). Must match the
        CASTEP k-path convention.

    Attributes
    ----------
    number_of_branches
        Number of phonon branches (``3 * n_atoms``).
    filename
        Input file path as provided.
    kpoints
        Number of q-points parsed from the file.
    frequencies
        Frequencies array of shape ``(kpoints, number_of_branches)`` in THz.
    kpath
        q-point coordinates array of shape ``(kpoints, 3)``.
    kpath_idx
        Per-segment q-point index lists (only set if ``kpath_in`` was given).
    """

    SEGMENT_TOL = 1e-4

    def __init__(self, castep_file: str, kpath_in: Any | None = None) -> None:
        """
        Initialise the reader and parse frequencies and q-point coordinates.

        Parameters
        ----------
        castep_file
            Path to a CASTEP output file readable by ASE.
        kpath_in
            Optional k-path (high-symmetry points) used to split the parsed
            q-points into per-segment index lists.
        """
        self.filename = castep_file

        if not castep_file:
            raise ValueError("castep_file must be provided.")

        atoms = ase.io.read(castep_file)
        self.number_of_branches = len(atoms) * 3

        self.read_in_file()
        self.get_frequencies()
        self.get_kpath()

        if kpath_in is not None:
            self.find_index(np.array(kpath_in, dtype=float))

        delattr(self, "filelines")

    def __str__(self) -> str:
        """
        Return a short description.

        Returns
        -------
        str
            Description string.
        """
        return "Phonon Dispersion (THz) from CASTEP file object"

    def read_in_file(self) -> None:
        """Read the input file into memory as lines."""
        with Path(self.filename).open(encoding="utf8") as handle:
            self.filelines = handle.readlines()

    def get_frequencies(self) -> None:
        """Parse phonon frequencies (THz) into ``self.frequencies``."""
        headlines = 2  # number of lines before frequency numbers appear

        thz_blocks = [
            self.filelines[i + headlines : i + headlines + self.number_of_branches]
            for i, val in enumerate(self.filelines)
            if re.search(r" \(THz\) ", val) is not None
        ]

        thz_lines = [line for block in thz_blocks for line in block]
        thz_vals = [line.split()[2] for line in thz_lines]

        frequencies = np.array(thz_vals, dtype=float)
        self.kpoints = int(len(frequencies) / self.number_of_branches)
        self.frequencies = np.reshape(
            frequencies,
            (self.kpoints, self.number_of_branches),
        )

    def get_kpath(self) -> None:
        """Parse q-point coordinates into ``self.kpath``."""
        qpt_lines = [
            self.filelines[i]
            for i, val in enumerate(self.filelines)
            if re.search(r"q-pt=", val) is not None
        ]

        qpts: list[list[str]] = []
        for line in qpt_lines:
            temp = line.split()[4:7]
            temp[2] = temp[2].replace(")", "")
            qpts.append(temp)

        self.kpath = np.array(qpts, dtype=float)

    def find_index(self, in_path: np.ndarray) -> None:
        """
        Split the parsed q-path into per-segment index lists (``kpath_idx``).

        Parameters
        ----------
        in_path
            Array of target high-symmetry points with shape ``(n_points, 3)``.
        """
        j = 0
        sympoint_idx: list[int] = []
        self.kpath_idx: list[list[int]] = []

        for i, val in enumerate(self.kpath):
            if abs(np.linalg.norm(in_path[j] - val)) < self.SEGMENT_TOL:
                sympoint_idx.append(i)
                j += 1

        for i in range(len(sympoint_idx) - 1):
            idxs = list(range(sympoint_idx[i], sympoint_idx[i + 1] + 1))
            self.kpath_idx.append(idxs)
