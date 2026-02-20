"""
Module for calculating thermal conductivity using Phono3py.

Code is adapted from https://github.com/MPA2suite/k_SRME/blob/6ff4c867/k_srme/conductivity.py
by Balázs Póta, Paramvir Ahlawat, Gábor Csányi, Michele Simoncelli
and https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/phonons/thermal_conductivity.py
by Janosh Riebesell.
See https://arxiv.org/abs/2408.00755 for details.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from pathlib import Path
import re
import sys
import traceback
from typing import Any
import warnings

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.utils import atoms_to_spglib_cell
import h5py
import numpy as np
import pandas as pd
from phono3py.api_phono3py import Phono3py
from phono3py.conductivity.wigner_rta import ConductivityWignerRTA
from phonopy.structure.atoms import PhonopyAtoms
import spglib
from tqdm import tqdm


# Backport-ish "StrEnum" behavior that works in Python 3.11
class StrEnumCompat(str, Enum):
    """
    Enum members behave like their underlying string values.

    - hash(TCKeys.mat_id) == hash("material_id")
    - TCKeys.mat_id == "material_id"
    """

    def __str__(self) -> str:
        """
        Return the underlying string value.

        Returns
        -------
        str
            The underlying string value.
        """
        return str(self.value)

    def __hash__(self) -> int:
        """
        Return the hash of the underlying string value.

        Returns
        -------
        int
            The hash of the underlying string value.
        """
        return hash(self.value)

    def __eq__(self, other) -> bool:
        """
        Compare against another enum or string value.

        Parameters
        ----------
        other : Any
            The value to compare against.

        Returns
        -------
        bool
            True if the values are equal, otherwise False.
        """
        if isinstance(other, Enum):
            return self.value == other.value
        return self.value == other

    def __format__(self, format_spec: str) -> str:
        """
        Format the underlying string value.

        Parameters
        ----------
        format_spec : str
            The format specification.

        Returns
        -------
        str
            The formatted string value.
        """
        return format(self.value, format_spec)


# Use the real StrEnum on 3.11+, otherwise the compat one
if sys.version_info >= (3, 11):
    from enum import StrEnum as _BaseStrEnum
else:
    _BaseStrEnum = StrEnumCompat


class TCKeys(_BaseStrEnum):
    """Keys for thermal conductivity dictionary."""

    kappa_tot_rta = "kappa_tot_rta"
    kappa_tot_avg = "kappa_tot_avg"
    kappa_p_rta = "kappa_p_rta"
    kappa_c = "kappa_c"
    mode_weights = "mode_weights"
    q_points = "q_points"
    ph_freqs = "ph_freqs"
    mode_kappa_tot_rta = "mode_kappa_tot_rta"
    mode_kappa_tot_avg = "mode_kappa_tot_avg"
    has_imag_ph_modes = "has_imag_ph_modes"
    temperatures = "temperatures"
    mat_id = "material_id"
    formula = "formula"
    heat_capacity = "heat_capacity"
    spg_num = "spg_num"
    init_spg_num = "initial_spg_num"
    relaxed_space_group_number = "relaxed_space_group_number"
    final_spg_num = "final_spg_num"
    srd = "srd"
    sre = "sre"
    srme = "srme"
    true_kappa_tot_avg = "true_kappa_tot_avg"
    name = "name"
    stability = "stability"


def calculate_fc2_set(
    ph3: Phono3py, calculator: Calculator, pbar_kwargs: dict[str, Any] | None = None
) -> np.ndarray:
    """
    Calculate 2nd order force constants.

    Requires initializing Phono3py with an FC2 supercell matrix.

    Parameters
    ----------
    ph3 : Phono3py
        Phono3py object for which to calculate force constants.
    calculator : Calculator
        ASE calculator to compute forces.
    pbar_kwargs : dict[str, Any] | None
        Arguments passed to tqdm progress bar.
        Defaults to None.

    Returns
    -------
    np.ndarray
        Array of forces for each displacement.
    """
    # print(f"Computing FC2 force set in {ph3.unitcell.formula}.")

    forces: list[np.ndarray] = []
    n_atoms = len(ph3.phonon_supercell)

    displacements = ph3.phonon_supercells_with_displacements
    for supercell in tqdm(
        displacements,
        desc=f"FC2 calculation: {ph3.unitcell.formula}",
        **pbar_kwargs or {},
    ):
        if supercell is not None:
            atoms = Atoms(
                supercell.symbols,
                cell=supercell.cell,
                positions=supercell.positions,
                pbc=True,
            )
            atoms.calc = calculator
            force = atoms.get_forces()
        else:
            force = np.zeros((n_atoms, 3))
        forces += [force]

    force_set = np.array(forces)
    ph3.phonon_forces = force_set
    return force_set


def calculate_fc3_set(
    ph3: Phono3py,
    calculator: Calculator,
    pbar_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Calculate 3rd order force constants.

    Parameters
    ----------
    ph3 : Phono3py
        Phono3py object for which to calculate force constants.
    calculator : Calculator
        ASE calculator to compute forces.
    pbar_kwargs : dict[str, Any] | None
        Passed to tqdm progress bar.
        Defaults to None.

    Returns
    -------
    np.ndarray
        Array of forces for each displacement.
    """
    forces: list[np.ndarray] = []
    n_atoms = len(ph3.supercell)

    desc = f"FC3 calculation: {ph3.unitcell.formula}"
    task_idx = (pbar_kwargs or {}).get("position")
    if task_idx:
        desc = f"{task_idx}. {desc}"
    displacements = ph3.supercells_with_displacements
    for supercell in tqdm(displacements, desc=desc, **pbar_kwargs or {}):
        if supercell is None:
            forces += [np.zeros((n_atoms, 3))]
        else:
            atoms = Atoms(
                supercell.symbols,
                cell=supercell.cell,
                positions=supercell.positions,
                pbc=True,
            )
            atoms.calc = calculator
            forces += [atoms.get_forces()]

    force_set = np.array(forces)
    ph3.forces = force_set
    return force_set


def init_phono3py(
    atoms: Atoms,
    *,
    fc2_supercell: np.ndarray,
    fc3_supercell: np.ndarray,
    q_point_mesh: tuple[int, int, int] = (20, 20, 20),
    displacement_distance: float = 0.03,
    symprec: float = 1e-5,
    **kwargs: Any,
) -> Phono3py:
    """
    Initialize Phono3py object from ASE Atoms.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object to initialize from.
    fc2_supercell : np.ndarray
        Supercell matrix for 2nd order force constants.
    fc3_supercell : np.ndarray
        Supercell matrix for 3rd order force constants.
    q_point_mesh : tuple[int, int, int]
        Mesh size for q-point sampling. Defaults
        to (20, 20, 20).
    displacement_distance : float
        Displacement distance for force calculations.
        Defaults to 0.03.
    symprec : float
        Symmetry precision for finding space group. Defaults to 1e-5.
    **kwargs : Any
        Passed to Phono3py constructor.

    Returns
    -------
    Phono3py
        Initialized Phono3py object.

    Raises
    ------
    ValueError
        If required metadata is missing from atoms.info.
    """
    unit_cell = PhonopyAtoms(atoms.symbols, cell=atoms.cell, positions=atoms.positions)
    ph3 = Phono3py(
        unitcell=unit_cell,
        supercell_matrix=fc3_supercell,
        phonon_supercell_matrix=fc2_supercell,
        primitive_matrix="auto",
        symprec=symprec,
        **kwargs,
    )
    ph3.mesh_numbers = q_point_mesh

    ph3.generate_displacements(distance=displacement_distance)

    return ph3


def get_fc2_and_freqs(
    ph3: Phono3py, calculator: Calculator, pbar_kwargs: dict[str, Any] | None = None
) -> tuple[Phono3py, np.ndarray, np.ndarray]:
    """
    Calculate 2nd order force constants and phonon frequencies.

    Parameters
    ----------
    ph3 : Phono3py
        Phono3py object for which to calculate force constants.
    calculator : Calculator
        ASE calculator to compute forces.
    pbar_kwargs : dict[str, Any] | None
        Arguments passed to tqdm progress bar.
        Defaults to None.

    Returns
    -------
    tuple[Phono3py, np.ndarray, np.ndarray]
        Tuple of (Phono3py object, force
        constants array, frequencies array [shape: (n_bz_grid, n_bands)]).

    Raises
    ------
    ValueError
        If mesh_numbers not set.
    """
    if ph3.mesh_numbers is None:
        raise ValueError(
            "mesh_numbers was not found in phono3py object and was not provided as "
            "an argument when calculating phonons from phono3py object."
        )

    pbar_kwargs = {"leave": False} | (pbar_kwargs or {})
    fc2_set = calculate_fc2_set(ph3, calculator, pbar_kwargs=pbar_kwargs)

    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_phonon_solver()

    freqs, _eigvecs, _grid = ph3.get_phonon_data()

    return ph3, fc2_set, freqs


def load_force_sets(
    ph3: Phono3py, fc2_set: np.ndarray, fc3_set: np.ndarray
) -> Phono3py:
    """
    Load pre-computed force sets into Phono3py object.

    Parameters
    ----------
    ph3 : Phono3py
        Phono3py object to load force sets into.
    fc2_set : np.ndarray
        2nd order force constants array.
    fc3_set : np.ndarray
        3rd order force constants array.

    Returns
    -------
    Phono3py
        Phono3py object with loaded force sets.
    """
    ph3.phonon_forces = fc2_set
    ph3.forces = fc3_set
    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.produce_fc3(symmetrize_fc3r=True)

    return ph3


def calculate_conductivity(
    ph3: Phono3py,
    temperatures: Sequence[float],
    boundary_mfp: float = 1e6,
    mode_kappa_thresh: float = 1e-6,
    **kwargs: Any,
) -> tuple[Phono3py, dict[str, np.ndarray], ConductivityWignerRTA]:
    """
    Calculate thermal conductivity.

    Parameters
    ----------
    ph3 : Phono3py
        Phono3py object for which to calculate conductivity.
    temperatures : list[float]
        Temperatures to compute conductivity at in Kelvin.
    boundary_mfp : float
        Mean free path in micrometer to calculate simple boundary
        scattering contribution to thermal conductivity. Defaults to 1e6.
    mode_kappa_thresh : float
        Threshold for mode kappa consistency check. Defaults
        to 1e-6.
    **kwargs : Any
        Passed to Phono3py.run_thermal_conductivity().

    Returns
    -------
    tuple[Phono3py, dict[str, np.ndarray], ConductivityWignerRTA]
        (Phono3py object,
        conductivity dict, conductivity object).
    """
    ph3.init_phph_interaction(symmetrize_fc3q=False)

    ph3.run_thermal_conductivity(
        **kwargs,
        temperatures=temperatures,
        is_isotope=True,
        # use type="wigner" to include both wave-like coherence (kappa_c) and
        # particle-like (kappa_p) conductivity contributions
        conductivity_type="wigner",
        boundary_mfp=boundary_mfp,
    )

    kappa = ph3.thermal_conductivity

    kappa_dict = {
        TCKeys.kappa_tot_rta: deepcopy(kappa.kappa_TOT_RTA[0]),
        TCKeys.kappa_p_rta: deepcopy(kappa.kappa_P_RTA[0]),
        TCKeys.kappa_c: deepcopy(kappa.kappa_C[0]),
        TCKeys.mode_weights: deepcopy(kappa.grid_weights),
        TCKeys.q_points: deepcopy(kappa.qpoints),
        TCKeys.ph_freqs: deepcopy(kappa.frequencies),
    }
    mode_kappa_total = kappa_dict[TCKeys.mode_kappa_tot_rta] = calc_mode_kappa_tot(
        deepcopy(kappa.mode_kappa_P_RTA[0]),
        deepcopy(kappa.mode_kappa_C[0]),
        deepcopy(kappa.mode_heat_capacities),
    )

    sum_mode_kappa_tot = mode_kappa_total.sum(
        axis=tuple(range(1, mode_kappa_total.ndim - 1))
    ) / np.sum(kappa_dict[TCKeys.mode_weights])

    kappa_tot_rta = kappa_dict[TCKeys.kappa_tot_rta]
    if np.any(np.abs(sum_mode_kappa_tot - kappa_tot_rta) > mode_kappa_thresh):
        warnings.warn(
            f"Total mode kappa does not sum to total kappa. {sum_mode_kappa_tot=}, "
            f"{kappa_tot_rta=}",
            stacklevel=2,
        )

    return ph3, kappa_dict, kappa


def calc_mode_kappa_tot(
    mode_kappa_p_rta: np.ndarray,
    mode_kappa_coherence: np.ndarray,
    heat_capacity: np.ndarray,
) -> np.ndarray:
    """
    Calculate total mode kappa from particle-like RTA and coherence terms.

    Parameters
    ----------
    mode_kappa_p_rta : np.ndarray
        Mode kappa from particle-like RTA with shape
        (T, q-points, bands, xyz).
    mode_kappa_coherence : np.ndarray
        Mode kappa from wave-like coherence with
        shape (T, q-points, bands, bands, xyz).
    heat_capacity : np.ndarray
        Mode heat capacities with shape
        (T, q-points, bands).

    Returns
    -------
    np.ndarray
        Total (particle-like + wave-like) thermal conductivity per phonon
        mode with shape (T, q-points, bands).
    """
    # Temporarily silence divide warnings since we handle NaN values below
    with np.errstate(divide="ignore", invalid="ignore"):
        mode_kappa_c_per_mode = 2 * (  # None equiv to np.newaxis
            (mode_kappa_coherence * heat_capacity[:, :, :, None, None])
            / (heat_capacity[:, :, :, None, None] + heat_capacity[:, :, None, :, None])
        ).sum(axis=2)

    mode_kappa_c_per_mode[np.isnan(mode_kappa_c_per_mode)] = 0

    return mode_kappa_c_per_mode + mode_kappa_p_rta


def check_imaginary_freqs(frequencies: np.ndarray, threshold: float = -0.01) -> bool:
    """
    Check if frequencies are imaginary.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies to check.
    threshold : float
        Threshold for imaginary frequencies. Defaults to -0.01.

    Returns
    -------
    bool
        True if imaginary frequencies are found.
    """
    # Return True if all frequencies are NaN, indicating invalid or missing data
    if np.all(pd.isna(frequencies)):
        return True

    # Check for imaginary frequencies in non-acoustic modes at gamma point (q=0)
    # Indices 3+ correspond to optical modes which should never be negative
    if np.any(frequencies[0, 3:] < 0):
        return True

    # Check acoustic modes at gamma point against threshold. First 3 modes at q=0
    # are acoustic and may be slightly negative due to numerical noise
    if np.any(frequencies[0, :3] < threshold):
        return True

    # Check for imaginary frequencies at any q-point except gamma
    # All frequencies should be positive away from gamma point
    return bool(np.any(frequencies[1:] < 0))


def get_spacegroup_number_from_atoms(atoms: Atoms, symprec: float = 1e-5) -> int:
    """
    Get space group number from ASE Atoms object.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object to analyze.
    symprec : float
        Symmetry precision. Defaults to 1e-5.

    Returns
    -------
    int
        Space group number.
    """
    dataset = spglib.get_symmetry_dataset(atoms_to_spglib_cell(atoms), symprec=symprec)
    return dataset.number


def calculate_kappa_avg(kappa: np.ndarray) -> np.ndarray:
    """
    Calculate directionally averaged trace of the conductivity tensor.

    Takes a thermal conductivity tensor and returns its trace (average of diagonal
    components). This represents the average thermal conductivity in the 3 spatial
    directions, which is a useful scalar metric for comparing materials.

    Parameters
    ----------
    kappa : np.ndarray
        Thermal conductivity tensor, typically of shape (..., 3, 3) where
        the last two dimensions represent the 3x3 conductivity tensor.
        Earlier dimensions may include temperatures or other parameters.

    Returns
    -------
    np.ndarray
        Average conductivity value(s). Returns np.nan if the input contains
        any NaN values or if the calculation fails. For multiple temperatures,
        returns an array of averages.
    """
    if np.any(pd.isna(kappa)):
        return np.array([np.nan])
    try:
        return np.asarray(kappa)[..., :3].mean(axis=-1)
    except Exception:
        warnings.warn(
            f"Failed to calculate kappa_avg: {traceback.format_exc()}", stacklevel=2
        )
        return np.array([np.nan])


_GRID_RE = re.compile(r"Grid point\s+\d+\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)")


class _TqdmIntercept:
    """
    Internal class to intercept stdout and update a tqdm progress bar.

    Based on output lines. Not intended for external use.

    Parameters
    ----------
    pbar : tqdm.tqdm
        The tqdm progress bar instance to update based on intercepted output.
    """

    def __init__(self, pbar):
        """
        Initialize the interceptor with a tqdm progress bar.

        Parameters
        ----------
        pbar : tqdm.tqdm
            The tqdm progress bar instance to update based on intercepted output.
        """
        self.pbar = pbar
        self._buf = ""

    def write(self, s: str) -> int:
        """
        Intercept writes to stdout, parse for progress, and update the tqdm bar.

        This method looks for lines in the output matching the pattern
        "Grid point X/Y" and updates the tqdm progress bar accordingly.
        It handles partial lines and ensures that the bar is updated smoothly
        without breaking the rendering.

        Parameters
        ----------
        s : str
            String to write, typically a line of output from the conductivity
            calculation.

        Returns
        -------
        int
            The number of characters written (length of the input string).
        """
        if not s:
            return 0
        self._buf += s

        while True:
            nl = self._buf.find("\n")
            if nl < 0:
                break
            line = self._buf[:nl]
            self._buf = self._buf[nl + 1 :]

            m = _GRID_RE.search(line)
            if not m:
                continue

            cur = int(m.group(1))
            total = int(m.group(2))

            # Learn total from the output
            if self.pbar.total is None or self.pbar.total != total:
                self.pbar.total = total

            # Start bar at 0 when cur==1 (or when the first seen cur is 2, show 1, etc.)
            if cur >= total:
                new_n = total
            else:
                new_n = max(0, min(total, cur - 1))
            if new_n != self.pbar.n:
                self.pbar.n = new_n
                self.pbar.refresh()

        return len(s)

    def flush(self):
        """
        Flush the buffer and update the progress bar to completion if needed.

        This should be called when the progress is complete to ensure the bar finished.
        """
        if self._buf:
            self.write("\n")
        if self.pbar.n < self.pbar.total:
            self.pbar.n = self.pbar.total
            self.pbar.refresh()

    # def isatty(self):
    #     return False

    @property
    def encoding(self) -> str:
        """
        Return the encoding of the underlying real stdout.

        Or 'utf-8' if it cannot be determined.

        Returns
        -------
        str
            Encoding of the underlying real stdout, or 'utf-8' if it cannot be
            determined.
        """
        return getattr(sys.__stdout__, "encoding", "utf-8")


@contextmanager
def tqdm_gridpoints(desc="Grid points", intercept_stderr=False, leave=False):
    """
    Context manager that creates a ``tqdm`` progress bar for grid-point iteration.

    Context manager that creates a ``tqdm`` progress bar for grid-point iteration
    and temporarily redirects standard output (and optionally standard error) so
    printed messages are routed through the bar without breaking its rendering.

    Parameters
    ----------
    desc : str, optional
        Description shown next to the progress bar. Default is ``"Grid points"``.
    intercept_stderr : bool, optional
        If ``True``, temporarily redirect ``sys.stderr`` to the same interceptor as
        ``sys.stdout`` while inside the context. Default is ``False``.
    leave : bool, optional
        Whether to leave the progress bar displayed after completion, passed to
        ``tqdm``. Default is ``False``.

    Yields
    ------
    tqdm.tqdm
        The active progress bar instance, allowing manual updates and metadata
        changes within the context.
    """
    old_out, old_err = sys.stdout, sys.stderr
    real_out = sys.__stdout__ if sys.__stdout__ is not None else old_out

    with tqdm(
        total=None, desc=desc, file=real_out, dynamic_ncols=True, leave=leave
    ) as pbar:
        interceptor = _TqdmIntercept(pbar)
        try:
            sys.stdout = interceptor
            if intercept_stderr:
                sys.stderr = interceptor
            yield pbar
        finally:
            interceptor.flush()
            sys.stdout = old_out
            sys.stderr = old_err


def dict_to_hdf5(
    data: dict[str, Any], h5file: h5py.File, group_name: str | None = None
) -> None:
    """
    Save a dictionary to an HDF5 file.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary to save.
    h5file : h5py.File
        Open HDF5 file object.
    group_name : str
        Optional name of the group to save the data under. Defaults to None.
    """
    if group_name is None:
        group = h5file
    else:
        group = h5file.require_group(group_name)

    for key, value in data.items():
        if isinstance(value, dict):
            dict_to_hdf5(value, group, group_name=f"{key}")
        else:
            if value is None:
                value = np.array(
                    [np.nan]
                )  # h5py does not support None, use NaN instead
            try:
                group.create_dataset(key, data=value)
            except TypeError as e:
                print(
                    f"Error: Could not save key '{key}' to HDF5 file. Unsupported "
                    f"type: {type(value)}, with value: {value}"
                )
                # traceback.print_exc()
                raise e


def hdf5_to_dict(h5file: h5py.File, group_name: str | None = None) -> dict[str, Any]:
    """
    Load a dictionary from an HDF5 file.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file object.
    group_name : str
        Optional name of the group to load the data from. Defaults to None.

    Returns
    -------
    dict[str, Any]
        Loaded dictionary.
    """
    if group_name is None:
        group = h5file
    else:
        group = h5file[group_name]

    data: dict[str, Any] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data[key] = hdf5_to_dict(h5file, group_name=f"{group.name}/{key}")
        else:
            data[key] = item[()]

    return data


def load_hdf5_subdir_dicts(
    path: Path | str, filename: str
) -> dict[str, dict[str, Any]]:
    """
    Load dictionaries from HDF5 files in subdirectories.

    Parameters
    ----------
    path : str | Path
        Path to the directory containing subdirectories with HDF5 files.
        filename (str): Name of the HDF5 file to load from each subdirectory.
    filename : str
        Name of the HDF5 file to load from each subdirectory.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary with subdirectory names as keys and loaded dictionaries as values.
    """
    subdirs = [d for d in Path(path).iterdir() if d.is_dir()]
    dicts = {}
    for d in subdirs:
        try:
            with h5py.File(d / filename, "r") as f:
                dicts[d.name] = hdf5_to_dict(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {d / filename}") from None
    return dicts
