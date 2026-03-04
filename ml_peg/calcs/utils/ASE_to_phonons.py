"""ASE → phonopy helpers for band structures and DOS/PDOS."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import ase
import ase.io
import numpy as np
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms


class AtomsToPhonons:
    """
    Compute phonon band structures from an ASE/phonopy workflow.

    Parameters
    ----------
    primitive_cell
        Primitive cell as an ASE ``Atoms`` or a ``PhonopyAtoms``-like object.
    phonon_grid
        Phonopy supercell matrix.
    displacement
        Displacement magnitude for finite-difference force constants.
    kpath
        K-path provided to phonopy's band structure routine.
    calculator
        Either an ASE calculator (default workflow) or an iterable of CASTEP files
        (when ``castep=True``).
    kpoints
        Number of points per k-path segment.
    castep
        If ``True``, forces are read from CASTEP output structures.
    plusminus
        Whether to generate plus/minus displacements.
    diagonal
        Whether to include diagonal displacements.

    Attributes
    ----------
    phonon
        Phonopy object holding force constants and band structure.
    supercells
        Displaced supercells used for force evaluation.
    forces
        List of forces arrays (one per displaced supercell).
    frequencies
        Concatenated frequencies along the k-path.
    xticks
        Tick positions corresponding to k-path segment boundaries.
    normal_ticks
        Evenly spaced tick positions used by some plotting utilities.
    """

    def __init__(
        self,
        primitive_cell: Any,
        phonon_grid: Any,
        displacement: float,
        kpath: Any,
        calculator: Any,
        kpoints: int = 100,
        castep: bool = False,
        plusminus: bool = False,
        diagonal: bool = True,
    ) -> None:
        """
        Initialise the workflow and compute the phonon band structure.

        Parameters
        ----------
        primitive_cell
            Primitive cell structure.
        phonon_grid
            Phonopy supercell matrix.
        displacement
            Displacement magnitude for finite differences.
        kpath
            K-path definition compatible with phonopy band structure routines.
        calculator
            ASE calculator or iterable of CASTEP output paths (when ``castep=True``).
        kpoints
            Number of points per k-path segment.
        castep
            If ``True``, read forces from CASTEP outputs instead of computing with ASE.
        plusminus
            Whether to generate ± displacements.
        diagonal
            Whether to generate diagonal displacements.
        """
        self.calculator_string = calculator
        self.get_supercell(
            primitive_cell, phonon_grid, displacement, plusminus, diagonal
        )

        self.forces: list[np.ndarray] | None = None
        if castep is False:
            self.get_forces_model()
        if castep is True:
            self.get_forces_castep()

        if self.forces is None or len(self.forces) == 0:
            raise RuntimeError("Forces were not generated.")

        self.get_band_struct(kpath, kpoints)

    def _scaled_positions(self, obj: Any) -> Any:
        """
        Return scaled positions from either PhonopyAtoms or ASE Atoms.

        Parameters
        ----------
        obj
            Object providing either a ``scaled_positions`` attribute or a
            ``get_scaled_positions()`` method.

        Returns
        -------
        Any
            Scaled positions array.
        """
        if hasattr(obj, "scaled_positions"):
            return obj.scaled_positions
        return obj.get_scaled_positions()

    def get_supercell(
        self,
        primitive_cell: Any,
        phonon_grid: Any,
        displacement: float,
        plusminus: bool,
        diagonal: bool,
    ) -> None:
        """
        Construct phonopy object and displaced supercells.

        Parameters
        ----------
        primitive_cell
            Primitive cell structure.
        phonon_grid
            Phonopy supercell matrix.
        displacement
            Displacement magnitude for finite differences.
        plusminus
            Whether to use ± displacements.
        diagonal
            Whether to generate diagonal displacements.
        """
        self.unitcell = PhonopyAtoms(
            symbols=primitive_cell.symbols
            if hasattr(primitive_cell, "symbols")
            else primitive_cell.get_chemical_symbols(),
            cell=primitive_cell.cell,
            scaled_positions=self._scaled_positions(primitive_cell),
        )

        self.phonon = Phonopy(self.unitcell, phonon_grid)
        self.phonon.generate_displacements(
            distance=displacement,
            is_plusminus=plusminus,
            is_diagonal=diagonal,
        )
        self.supercells = self.phonon.supercells_with_displacements

    def get_forces_model(self) -> None:
        """Compute forces using an ASE calculator."""
        potential = self.calculator_string
        self.forces = []
        self.atoms: list[ase.Atoms] = []

        for s in self.supercells:
            atoms = ase.Atoms(
                symbols=list(s.symbols),
                cell=s.cell,
                scaled_positions=self._scaled_positions(s),
                pbc=True,
            )
            self.atoms.append(atoms)
            atoms.calc = potential
            self.forces.append(atoms.get_forces())

    def get_forces_castep(self) -> None:
        """Read forces from CASTEP output structures."""
        forces: list[np.ndarray] = []
        self.atoms = []

        for _i, path in enumerate(self.calculator_string):
            castep_atoms = ase.io.read(path)
            self.atoms.append(castep_atoms)
            forces.append(castep_atoms.get_forces())

        self.forces = forces

    def get_band_struct(self, kpath: Any, kpoints: int) -> None:
        """
        Compute band structure along the provided k-path.

        Parameters
        ----------
        kpath
            K-path definition compatible with
            ``get_band_qpoints_and_path_connections``.
        kpoints
            Number of points per k-path segment.
        """
        self.phonon.forces = np.asarray(self.forces, dtype=float)
        self.phonon.produce_force_constants()

        qpoints, connections = get_band_qpoints_and_path_connections(
            kpath, npoints=kpoints
        )
        self.phonon.run_band_structure(
            qpoints,
            with_eigenvectors=True,
            path_connections=connections,
        )

        bs = self.phonon.get_band_structure_dict()
        self.frequencies_array = bs["frequencies"]
        self.eigenvectors = bs["eigenvectors"]

        self.frequencies = np.array(self.frequencies_array[0], copy=True)

        xticks: list[int] = []
        x = 0
        for val in self.frequencies_array[1:]:
            self.frequencies = np.append(self.frequencies, val, axis=0)
            x += len(val)
            xticks.append(x)

        self.xticks = xticks
        n_kpoints = kpoints * len(kpath[0])
        n_trace = int(n_kpoints / (len(kpath[0])))
        self.normal_ticks = [i * n_trace for i in range(len(kpath[0]))]


class AtomsToPDOS:
    """
    Compute force constants and DOS/PDOS/thermal properties via phonopy.

    Parameters
    ----------
    primitive_cell
        Primitive cell as an ASE ``Atoms`` or a ``PhonopyAtoms``-like object.
    phonon_grid
        Phonopy supercell matrix.
    displacement
        Displacement magnitude for finite-difference force constants.
    calculator
        Either an ASE calculator (default workflow) or an iterable of CASTEP files
        (when ``castep=True``).
    kpoints
        Unused (kept for API compatibility with :class:`AtomsToPhonons`).
    castep
        If ``True``, forces are read from CASTEP output structures.
    plusminus
        Whether to generate plus/minus displacements.
    diagonal
        Whether to include diagonal displacements.

    Attributes
    ----------
    phonon
        Phonopy object holding force constants and mesh results.
    supercells
        Displaced supercells used for force evaluation.
    forces
        List of forces arrays (one per displaced supercell).
    pdos
        Projected DOS dictionary (set by :meth:`get_pdos`).
    dos
        Total DOS dictionary (set by :meth:`get_dos`).
    tp_dict
        Thermal properties dictionary (set by :meth:`get_tp`).
    """

    def __init__(
        self,
        primitive_cell: Any,
        phonon_grid: Any,
        displacement: float,
        calculator: Any,
        kpoints: int = 100,
        castep: bool = False,
        plusminus: bool = False,
        diagonal: bool = True,
    ) -> None:
        """
        Initialise the workflow and build force constants.

        Parameters
        ----------
        primitive_cell
            Primitive cell structure.
        phonon_grid
            Phonopy supercell matrix.
        displacement
            Displacement magnitude for finite differences.
        calculator
            ASE calculator or iterable of CASTEP output paths (when ``castep=True``).
        kpoints
            Unused (kept for API compatibility).
        castep
            If ``True``, read forces from CASTEP outputs instead of computing with ASE.
        plusminus
            Whether to generate ± displacements.
        diagonal
            Whether to generate diagonal displacements.
        """
        _ = kpoints
        self.calculator_string = calculator
        self.get_supercell(
            primitive_cell, phonon_grid, displacement, plusminus, diagonal
        )

        self.forces: list[np.ndarray] | None = None
        if castep is False:
            self.get_forces_model()
        if castep is True:
            self.get_forces_castep()

        if self.forces is None:
            raise RuntimeError(
                "Forces were not generated. Check calculator/castep inputs."
            )

        self.get_dynamical_matrix()

    def _scaled_positions(self, obj: Any) -> Any:
        """
        Return scaled positions from either PhonopyAtoms or ASE Atoms.

        Parameters
        ----------
        obj
            Object providing either a ``scaled_positions`` attribute or a
            ``get_scaled_positions()`` method.

        Returns
        -------
        Any
            Scaled positions array.
        """
        return (
            obj.scaled_positions
            if hasattr(obj, "scaled_positions")
            else obj.get_scaled_positions()
        )

    def get_supercell(
        self,
        primitive_cell: Any,
        phonon_grid: Any,
        displacement: float,
        plusminus: bool,
        diagonal: bool,
    ) -> None:
        """
        Construct phonopy object and displaced supercells.

        Parameters
        ----------
        primitive_cell
            Primitive cell structure.
        phonon_grid
            Phonopy supercell matrix.
        displacement
            Displacement magnitude for finite differences.
        plusminus
            Whether to use ± displacements.
        diagonal
            Whether to generate diagonal displacements.
        """
        self.unitcell = PhonopyAtoms(
            symbols=primitive_cell.symbols
            if hasattr(primitive_cell, "symbols")
            else primitive_cell.get_chemical_symbols(),
            cell=primitive_cell.cell,
            scaled_positions=self._scaled_positions(primitive_cell),
        )
        self.phonon = Phonopy(self.unitcell, phonon_grid)
        self.phonon.generate_displacements(
            distance=displacement,
            is_plusminus=plusminus,
            is_diagonal=diagonal,
        )
        self.supercells = self.phonon.supercells_with_displacements

    def get_forces_model(self) -> None:
        """Compute forces using an ASE calculator."""
        potential = self.calculator_string
        self.forces = []
        self.atoms: list[ase.Atoms] = []

        for s in self.supercells:
            atoms = ase.Atoms(
                symbols=list(s.symbols),
                cell=s.cell,
                scaled_positions=self._scaled_positions(s),
                pbc=True,
            )
            self.atoms.append(atoms)
            atoms.calc = potential
            self.forces.append(atoms.get_forces())

    def get_forces_castep(self) -> None:
        """Read forces from CASTEP output structures."""
        forces: list[np.ndarray] = []
        self.atoms = []

        for _i, path in enumerate(self.calculator_string):
            castep_atoms = ase.io.read(path)
            self.atoms.append(castep_atoms)
            forces.append(castep_atoms.get_forces())

        self.forces = forces

    def get_dynamical_matrix(self) -> None:
        """
        Build force constants from the stored forces.

        Raises
        ------
        ValueError
            If the number of force sets does not match the number of supercells.
        """
        forces = np.asarray(self.forces, dtype=float)

        if len(forces) != len(self.supercells):
            raise ValueError(
                f"Number of force sets ({len(forces)}) != number of supercells "
                f"({len(self.supercells)})."
            )

        self.phonon.forces = forces
        self.phonon.produce_force_constants()

    def get_pdos(self, qmesh: Sequence[int]) -> None:
        """
        Compute projected DOS and total DOS on a mesh.

        Parameters
        ----------
        qmesh
            Q-mesh used for DOS calculations.
        """
        self.phonon.run_mesh(qmesh, with_eigenvectors=True, is_mesh_symmetry=False)
        self.phonon.run_projected_dos()
        self.phonon.run_total_dos()
        self.pdos = self.phonon.get_projected_dos_dict()

    def get_tp(
        self,
        qmesh: Sequence[int],
        tmin: float = 0,
        tmax: float = 2000,
        tstep: float = 100,
    ) -> None:
        """
        Compute thermal properties.

        Parameters
        ----------
        qmesh
            Q-mesh for thermal properties.
        tmin
            Minimum temperature (K).
        tmax
            Maximum temperature (K).
        tstep
            Temperature step (K).
        """
        self.phonon.run_mesh(qmesh)
        self.phonon.run_thermal_properties(t_step=tstep, t_max=tmax, t_min=tmin)
        self.tp_dict = self.phonon.get_thermal_properties_dict()

    def get_dos(self, qmesh: Sequence[int]) -> None:
        """
        Compute total DOS on a mesh.

        Parameters
        ----------
        qmesh
            Q-mesh used for total DOS calculation.
        """
        self.phonon.run_mesh(qmesh)
        self.phonon.run_total_dos()
        self.dos = self.phonon.get_total_dos_dict()
