"""
Ti64 phonons CASTEP phonon dispersion reader.

This module provides a lightweight parser for CASTEP phonon output files to
extract phonon frequencies (in THz) and q-point coordinates. Optionally, the
dispersion x-axis can be rescaled to span [0, 1] along a provided k-path.

Notes
-----
- This parser assumes the CASTEP output uses THz units for phonon frequencies.
- The rescaling assumes ``kpath_in`` matches the k-path used in the CASTEP run.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import warnings

import ase.io
import numpy as np

warnings.simplefilter("ignore")


class PhononFromCastep:
    """
    Extract phonon frequencies and k-points from a CASTEP phonon calculation.

    Parameters
    ----------
    castep_file
        Path to a CASTEP output file readable by ASE.
    kpath_in
        Optional k-path (high-symmetry points) used to rescale the dispersion
        axis onto [0, 1]. Must match the CASTEP k-path convention.
    verbose
        If ``True``, print basic debug information.

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
    xscale
        Optional rescaled x-axis (only present if ``kpath_in`` was provided).
    """

    RESCALE_TOL = 1e-4

    def __init__(
        self,
        castep_file: str,
        kpath_in: Any | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialise the reader and parse frequencies and q-point coordinates.

        Parameters
        ----------
        castep_file
            Path to a CASTEP output file readable by ASE.
        kpath_in
            Optional k-path (high-symmetry points) used to rescale the dispersion
            axis onto [0, 1]. Must match the CASTEP k-path convention.
        verbose
            If ``True``, print basic debug information.
        """
        self.filename = castep_file

        if not castep_file:
            raise ValueError("castep_file must be provided.")

        try:
            atoms = ase.io.read(castep_file)
        except AttributeError as exc:
            raise TypeError("Invalid input type for castep_file.") from exc

        self.number_of_branches = len(atoms) * 3

        self.read_in_file()
        self.get_frequencies()
        self.get_kpath()

        if verbose:
            print(f"Atoms object info:\n{atoms}\n")
            print(self.__dict__.keys(), "\n")

        if kpath_in is not None:
            self.rescale_xaxis(kpath_in)
            if verbose:
                print("k-path rescaled")
        elif verbose:
            print("no k-path re-scaling done")

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
        Find indices of high-symmetry points along the parsed k-path.

        Parameters
        ----------
        in_path
            Array of target high-symmetry points with shape ``(n_points, 3)``.
        """
        j = 0
        sympoint_idx: list[int] = []
        self.kpath_idx: list[list[int]] = []

        for i, val in enumerate(self.kpath):
            if abs(np.linalg.norm(in_path[j] - val)) < self.RESCALE_TOL:
                sympoint_idx.append(i)
                j += 1

        for i in range(len(sympoint_idx) - 1):
            idxs = list(range(sympoint_idx[i], sympoint_idx[i + 1] + 1))
            self.kpath_idx.append(idxs)

    def rescale_xaxis(self, rescale_xaxis: Any) -> None:
        """
        Rescale the dispersion axis to span [0, 1] over the provided k-path.

        Parameters
        ----------
        rescale_xaxis
            Iterable of high-symmetry points (each a length-3 coordinate) used
            to determine segment boundaries for rescaling.
        """
        in_path = np.array(rescale_xaxis, dtype=float)
        self.find_index(in_path)

        xsplit = 1.0 / len(self.kpath_idx)
        xscale = [0.0]
        pos = 0.0

        kpath_cut: list[list[int]] = [self.kpath_idx[0]]
        for i in range(len(self.kpath_idx) - 1):
            kpath_cut.append(self.kpath_idx[i + 1][1:])

        x_inc = xsplit / (len(kpath_cut[0]) - 1)
        for _ in range(len(kpath_cut[0]) - 1):
            pos += x_inc
            xscale.append(pos)

        for i in range(len(kpath_cut) - 1):
            x_inc = xsplit / len(kpath_cut[i + 1])
            for _ in range(len(kpath_cut[i + 1])):
                pos += x_inc
                xscale.append(pos)

        self.xscale = np.array(xscale, dtype=float)
