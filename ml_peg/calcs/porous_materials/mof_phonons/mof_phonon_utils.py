"""
Shared utilities for the MOF phonon benchmarks.

A single phonon calculation per framework underpins two scored quantities,
heat capacity and inelastic neutron scattering (INS), plus the fraction of
imaginary modes reported by both. Everything that both quantities need lives
here so that the force constants are only ever produced once.

References
----------
Force constants, thermal properties and mode counting use phonopy [1]_. The
default INS backend is Euphonic [2]_; the alternative backend is the Abins
algorithm [3]_ as distributed in Mantid.

.. [1] A. Togo and I. Tanaka, "First principles phonon calculations in
   materials science", Scripta Materialia 108, 1-5 (2015).
   https://doi.org/10.1016/j.scriptamat.2015.07.021
.. [2] R. L. Fair, A. J. Jackson, D. J. Voneshen, et al., "Euphonic:
   inelastic neutron scattering simulations from force constants and
   visualisation tools for phonon properties", Journal of Applied
   Crystallography 55, 1689-1703 (2022).
   https://doi.org/10.1107/S1600576722009256
.. [3] K. Dymkowski, S. F. Parker, F. Fernandez-Alonso and S. Mukhopadhyay,
   "AbINS: The modern software for INS interpretation", Physica B: Condensed
   Matter 551, 443-448 (2018). https://doi.org/10.1016/j.physb.2018.02.034
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import shutil
from typing import Any

from ase import Atoms
import numpy as np
from phonopy.api_phonopy import Phonopy

OUTPUT_PATH = Path(__file__).parent / "outputs"
# The calc step copies the reference files here so that analysis does not need
# to reach S3 itself.
REFERENCE_PATH = OUTPUT_PATH / "reference"

S3_KEY = "inputs/porous_materials/mof_phonons/mof_phonons.zip"
S3_FILENAME = "mof_phonons.zip"
# Point this at an unpacked copy of the benchmark data to work offline.
DATA_DIR_ENV = "ML_PEG_MOF_PHONONS_DATA"

HEAT_CAPACITY_REFERENCE_NAME = "heat_capacity_reference.json"
INS_REFERENCE_NAME = "ins_reference.json.gz"
BULK_MODULUS_REFERENCE_NAME = "bulk_modulus_reference.json"

# CODATA 2018 conversion between wavenumber and energy.
CM1_PER_MEV = 8.0655439
# phonopy reports frequencies in THz; 1 THz = 4.135667696 meV.
MEV_PER_THZ = 4.135667696
# phonopy excludes modes at or below this frequency from the thermal-property
# integration; zero makes that exclusion exactly the imaginary modes.
THERMAL_CUTOFF_FREQUENCY = 0.0


def get_data_dir() -> Path:
    """
    Get the directory holding the benchmark structures and reference data.

    Downloaded from the ML-PEG data bucket unless the environment variable
    named by ``DATA_DIR_ENV`` points at an unpacked copy.

    Returns
    -------
    Path
        Directory containing ``structures/`` and the reference files.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override)

    from ml_peg.calcs.utils.utils import download_s3_data

    return download_s3_data(key=S3_KEY, filename=S3_FILENAME) / "mof_phonons"


def get_structure_names(data_dir: Path) -> list[str]:
    """
    List the frameworks in the benchmark set.

    Parameters
    ----------
    data_dir
        Directory returned by :func:`get_data_dir`.

    Returns
    -------
    list[str]
        Sorted framework names, taken from the bundled CIF filenames.
    """
    return sorted(path.stem for path in (data_dir / "structures").glob("*.cif"))


def copy_reference_data(data_dir: Path) -> None:
    """
    Copy the reference files next to the calculation outputs.

    Parameters
    ----------
    data_dir
        Directory returned by :func:`get_data_dir`.
    """
    REFERENCE_PATH.mkdir(parents=True, exist_ok=True)
    for name in (
        HEAT_CAPACITY_REFERENCE_NAME,
        INS_REFERENCE_NAME,
        BULK_MODULUS_REFERENCE_NAME,
    ):
        source = data_dir / name
        if source.exists():
            shutil.copy2(source, REFERENCE_PATH / name)


def diagonal_supercell_matrix(
    atoms: Atoms, min_length: float = 20.0
) -> list[list[int]]:
    """
    Build the smallest diagonal supercell with all lattice vectors long enough.

    Parameters
    ----------
    atoms
        Unit cell to expand.
    min_length
        Minimum supercell lattice vector length in Angstrom.

    Returns
    -------
    list[list[int]]
        Diagonal 3x3 supercell matrix.
    """
    lengths = atoms.cell.lengths()
    reps = [max(1, int(np.ceil(min_length / length))) for length in lengths]
    return [
        [reps[0], 0, 0],
        [0, reps[1], 0],
        [0, 0, reps[2]],
    ]


def imaginary_mode_percentage(phonons: Phonopy) -> float:
    """
    Get the q-point-weighted percentage of imaginary modes on the phonon mesh.

    Defined as the modes phonopy leaves out of the thermal-property
    integration, as a fraction of all modes on the mesh::

        100 * (number_of_modes - number_of_integrated_modes) / number_of_modes

    Both counts are q-point-weighted, and phonopy integrates modes with
    frequency strictly greater than its cutoff, so with a cutoff of zero the
    excluded modes are exactly the imaginary ones. Tying the metric to the
    thermal-property bookkeeping means the heat capacity and this number
    describe the same set of modes.

    ``Phonopy.run_thermal_properties`` must already have been called, with
    ``cutoff_frequency`` left at its default of zero.

    Parameters
    ----------
    phonons
        Phonopy object whose thermal properties have been run.

    Returns
    -------
    float
        Percentage of imaginary modes, between 0 and 100.

    Notes
    -----
    When Gamma lies on the mesh its three acoustic branches are numerically
    zero and may land on either side of the strict cutoff. With symmetrised
    force constants they are typically very slightly positive and so are
    integrated, but if they come out non-positive they are excluded and add a
    small floor of three q-point weights out of the total.
    """
    thermal = phonons.thermal_properties
    total = thermal.number_of_modes
    if not total:
        return float("nan")
    excluded = total - thermal.number_of_integrated_modes
    return 100.0 * excluded / total


def heat_capacity_per_gram(phonons: Phonopy, temperature: float) -> float:
    """
    Get the isochoric heat capacity per gram at one temperature.

    ``Phonopy.run_thermal_properties`` must already have been called over a
    temperature range that contains ``temperature``. phonopy reports the heat
    capacity in J/K/mol of the cell it was run on, so dividing by that cell's
    molar mass gives J/g/K.

    Parameters
    ----------
    phonons
        Phonopy object with thermal properties available.
    temperature
        Temperature in K at which to report the heat capacity.

    Returns
    -------
    float
        Heat capacity in J/g/K, linearly interpolated onto ``temperature``.
    """
    thermal = phonons.get_thermal_properties_dict()
    temperatures = np.asarray(thermal["temperatures"], dtype=float)
    capacities = np.asarray(thermal["heat_capacity"], dtype=float)

    # phonopy's thermal properties refer to the primitive cell.
    molar_mass = float(np.sum(phonons.primitive.masses))
    return float(np.interp(temperature, temperatures, capacities) / molar_mass)


# Force constants produced from ASE forces are in eV/Angstrom^2, with masses
# in AMU and lengths in Angstrom. phonopy does not record this in the summary
# file it writes, and Euphonic would otherwise fall back to assuming it, so the
# block is written explicitly.
PHYSICAL_UNIT_BLOCK = (
    "physical_unit:\n"
    '  atomic_mass: "AMU"\n'
    '  length: "Angstrom"\n'
    '  force_constants: "eV/Angstrom^2"\n'
)

FORCE_CONSTANTS_FILENAME = "force_constants.hdf5"
SUMMARY_FILENAME = "phonopy.yaml"


def write_phonopy_inputs(phonons: Phonopy, directory: Path) -> Path:
    """
    Write the phonopy summary and force constants that the INS backends read.

    The force constants go to HDF5 rather than into the summary file: a MOF
    supercell has thousands of atoms, and serialising that force-constant
    array as YAML would produce a file of many gigabytes.

    Parameters
    ----------
    phonons
        Phonopy object with force constants computed.
    directory
        Directory to write into.

    Returns
    -------
    Path
        Path to the written summary file.
    """
    from phonopy.file_IO import write_force_constants_to_hdf5

    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / SUMMARY_FILENAME
    phonons.save(filename=summary, settings={"force_constants": False})
    # Top-level YAML keys are unordered, so the unit block can be prepended
    # without parsing the file back in.
    summary.write_text(PHYSICAL_UNIT_BLOCK + "\n" + summary.read_text())

    write_force_constants_to_hdf5(
        phonons.force_constants, filename=directory / FORCE_CONSTANTS_FILENAME
    )
    return summary


def ins_spectrum_euphonic(
    phonons: Phonopy,
    directory: Path,
    energy_bins_mev: np.ndarray,
    temperature: float,
    q_mesh: tuple[int, int, int],
) -> np.ndarray:
    """
    Compute a neutron-weighted vibrational spectrum with Euphonic.

    The spectrum is the incoherent-approximation neutron-weighted density of
    states, summed over atoms, as implemented by Euphonic's
    ``calculate_pdos(weighting="incoherent")``. This is the one-phonon powder
    INS intensity, and Euphonic is the pip-installable default backend.

    See Fair et al. (2022), reference [2] in the module docstring.

    Parameters
    ----------
    phonons
        Phonopy object with force constants computed.
    directory
        Scratch directory for the phonopy summary file.
    energy_bins_mev
        Energy bin edges in meV.
    temperature
        Temperature in K. Retained for interface symmetry with the Abins
        backend; the incoherent-approximation weighted DOS is temperature
        independent.
    q_mesh
        Monkhorst-Pack mesh used to sample the Brillouin zone.

    Returns
    -------
    np.ndarray
        Intensity on the bin centres of ``energy_bins_mev``, arbitrary units.
    """
    from euphonic import ForceConstants, ureg
    from euphonic.util import mp_grid

    del temperature  # Not used by the weighted-DOS backend.

    summary = write_phonopy_inputs(phonons, directory)
    force_constants = ForceConstants.from_phonopy(
        path=str(summary.parent),
        summary_name=summary.name,
        fc_name=FORCE_CONSTANTS_FILENAME,
    )

    modes = force_constants.calculate_qpoint_phonon_modes(
        mp_grid(q_mesh), asr="reciprocal"
    )
    pdos = modes.calculate_pdos(
        dos_bins=energy_bins_mev * ureg("meV"), weighting="incoherent"
    )
    return np.asarray(pdos.sum().y_data.magnitude, dtype=float)


def ins_spectrum_abins(
    phonons: Phonopy,
    directory: Path,
    energy_bins_mev: np.ndarray,
    temperature: float,
    q_mesh: tuple[int, int, int],
) -> np.ndarray:
    """
    Compute a TOSCA INS spectrum with Mantid's Abins algorithm.

    Used when Mantid is importable. Abins adds the instrument kinematic
    trajectory, second-order quantum events and autoconvolution, so it is a
    closer match to a measured TOSCA spectrum than the weighted DOS, but
    Mantid is conda-only and so cannot be a hard dependency.

    See Dymkowski et al. (2018), reference [3] in the module docstring. The
    algorithm settings mirror those used for the TOSCA spectra this benchmark
    is scored against.

    Parameters
    ----------
    phonons
        Phonopy object with force constants computed.
    directory
        Scratch directory for the phonopy summary file and Abins cache.
    energy_bins_mev
        Energy bin edges in meV, used to resample the Abins output.
    temperature
        Sample temperature in K.
    q_mesh
        Unused; Abins derives its own sampling from the force constants.

    Returns
    -------
    np.ndarray
        Intensity on the bin centres of ``energy_bins_mev``, arbitrary units.
    """
    from mantid.simpleapi import Abins, mtd

    del q_mesh  # Abins samples the Brillouin zone internally.

    summary = write_phonopy_inputs(phonons, directory)
    cache_dir = directory / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    workspace_name = f"abins_{directory.name}"
    Abins(
        VibrationalOrPhononFile=str(summary),
        AbInitioProgram="FORCECONSTANTS",
        OutputWorkspace=workspace_name,
        TemperatureInKelvin=str(temperature),
        SumContributions=True,
        ScaleByCrossSection="Total",
        QuantumOrderEventsNumber="2",
        Autoconvolution=True,
        Instrument="TOSCA",
        Setting="Backward (TOSCA)",
        CacheDirectory=str(cache_dir),
    )

    workspace = mtd[f"{workspace_name}_total"]
    edges = np.asarray(workspace.extractX(), dtype=float).flatten()
    intensity = np.asarray(workspace.extractY(), dtype=float).flatten()
    # Abins returns wavenumbers; convert to meV and resample onto our grid.
    centres_mev = 0.5 * (edges[:-1] + edges[1:]) / CM1_PER_MEV
    target = 0.5 * (energy_bins_mev[:-1] + energy_bins_mev[1:])
    return np.interp(target, centres_mev, intensity, left=0.0, right=0.0)


INS_BACKENDS = {
    "euphonic": ins_spectrum_euphonic,
    "abins": ins_spectrum_abins,
}


def resolve_ins_backend(requested: str = "auto") -> str:
    """
    Choose the INS backend to use.

    Parameters
    ----------
    requested
        ``"euphonic"``, ``"abins"``, or ``"auto"``. ``"auto"`` prefers Abins
        when Mantid is importable and falls back to Euphonic.

    Returns
    -------
    str
        Name of the selected backend.

    Raises
    ------
    ValueError
        If ``requested`` is not a known backend.
    ImportError
        If the requested backend is not installed.
    """
    if requested not in {*INS_BACKENDS, "auto"}:
        msg = (
            f"Unknown INS backend {requested!r}; expected one of {sorted(INS_BACKENDS)}"
        )
        raise ValueError(msg)

    from importlib.util import find_spec

    if requested == "auto":
        if find_spec("mantid") is not None:
            return "abins"
        if find_spec("euphonic") is not None:
            return "euphonic"
        msg = (
            "No INS backend available. Install Euphonic with "
            "`pip install ml-peg[ins]`, or provide a conda Mantid installation."
        )
        raise ImportError(msg)

    module = "mantid" if requested == "abins" else "euphonic"
    if find_spec(module) is None:
        msg = f"INS backend {requested!r} requires {module}, which is not installed."
        raise ImportError(msg)
    return requested


def _prepare_spectrum(
    energy: np.ndarray, intensity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean one spectrum for comparison.

    Intensities are clipped at zero, points are sorted by energy, and any
    repeated energies are collapsed to their mean intensity. Digitised
    spectra routinely contain both negative excursions and repeated
    abscissae, and ``numpy.interp`` requires a strictly increasing grid.

    Parameters
    ----------
    energy
        Energies in meV.
    intensity
        Intensities, arbitrary units.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted, deduplicated energies and their non-negative intensities.
    """
    energy = np.asarray(energy, dtype=float)
    intensity = np.clip(np.asarray(intensity, dtype=float), 0.0, None)

    order = np.argsort(energy, kind="stable")
    energy, intensity = energy[order], intensity[order]

    unique, inverse = np.unique(energy, return_inverse=True)
    if unique.size != energy.size:
        summed = np.bincount(inverse, weights=intensity)
        counts = np.bincount(inverse)
        return unique, summed / counts
    return energy, intensity


def wasserstein_spectrum_distance(
    ref_energy: np.ndarray,
    ref_intensity: np.ndarray,
    pred_energy: np.ndarray,
    pred_intensity: np.ndarray,
    window: tuple[float, float] | None = None,
) -> float:
    """
    Get the Wasserstein-1 distance between two vibrational spectra.

    INS intensities are in arbitrary units and the reference spectra are
    digitised from figures on their own irregular energy grids, so neither the
    intensity scale nor the sampling of the two spectra can be compared
    directly. Both are therefore restricted to a common energy window,
    resampled onto a single uniform grid spanning it, and normalised to unit
    area over that window, which turns each into a distribution over energy.
    Resampling matters: digitised grids are up to an order of magnitude denser
    in some regions than others, and weighting raw intensities on the native
    grid would count densely sampled regions more heavily.

    Normalising by area is what puts the measurement and the calculation on
    a common intensity footing: it is equivalent to scaling the measured
    trace until its area matches the calculated one. Any per-spectrum factor
    applied beforehand is therefore absorbed and cannot change the result.

    The Wasserstein-1 distance between those distributions measures how far
    spectral weight has to move, in meV, to turn one into the other, and so
    penalises systematically over- or under-stiff modes rather than
    differences in overall intensity.

    Computed with ``scipy.stats.wasserstein_distance``, which evaluates the
    Wasserstein-1 (earth mover's) distance between two weighted samples.

    Parameters
    ----------
    ref_energy
        Reference energies in meV.
    ref_intensity
        Reference intensities, arbitrary units.
    pred_energy
        Predicted energies in meV.
    pred_intensity
        Predicted intensities, arbitrary units.
    window
        Energy window in meV to compare over. When omitted, the overlap of
        the two energy ranges is used. Every reference shipped with this
        benchmark declares one, because the computed spectrum keeps full
        intensity in the C-H and O-H stretch bands near 370-470 meV while a
        measurement suppresses them through the Debye-Waller factor and the
        instrument's kinematic trajectory. Comparing over the full overlap
        therefore scores the missing suppression rather than the model.

    Returns
    -------
    float
        Wasserstein-1 distance in meV, or NaN if the spectra do not overlap
        or carry no intensity in the window.
    """
    from scipy.stats import wasserstein_distance

    # numpy renamed trapz to trapezoid in 2.0; ML-PEG supports both.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz

    ref_energy, ref_intensity = _prepare_spectrum(ref_energy, ref_intensity)
    pred_energy, pred_intensity = _prepare_spectrum(pred_energy, pred_intensity)

    if window is None:
        low = max(ref_energy.min(), pred_energy.min())
        high = min(ref_energy.max(), pred_energy.max())
    else:
        low, high = float(window[0]), float(window[1])
        low = max(low, ref_energy.min(), pred_energy.min())
        high = min(high, ref_energy.max(), pred_energy.max())

    if not np.isfinite([low, high]).all() or high <= low:
        return float("nan")

    ref_mask = (ref_energy >= low) & (ref_energy <= high)
    pred_mask = (pred_energy >= low) & (pred_energy <= high)
    if ref_mask.sum() < 2 or pred_mask.sum() < 2:
        return float("nan")

    # Resample onto one uniform grid, at the finer of the two native
    # resolutions, so the areas below are computed consistently.
    n_points = max(int(ref_mask.sum()), int(pred_mask.sum()))
    common_energy = np.linspace(low, high, n_points)
    ref_common = np.interp(common_energy, ref_energy[ref_mask], ref_intensity[ref_mask])
    pred_common = np.interp(
        common_energy, pred_energy[pred_mask], pred_intensity[pred_mask]
    )

    ref_area = float(trapezoid(ref_common, common_energy))
    pred_area = float(trapezoid(pred_common, common_energy))
    if ref_area <= 0 or pred_area <= 0:
        return float("nan")

    return float(
        wasserstein_distance(
            common_energy,
            common_energy,
            u_weights=ref_common / ref_area,
            v_weights=pred_common / pred_area,
        )
    )


# A framework that has been synthesised and measured is dynamically stable, so
# the physically correct number of imaginary modes is zero. There is no
# per-structure reference to compare against, so the metric is the mean
# deviation from that expected value.
PERCENT_IMAGINARY_REFERENCE = 0.0


def load_phonon_summaries(model_name: str) -> dict[str, dict[str, Any]]:
    """
    Load every phonon summary written for one model.

    Both scored quantities read these summaries, which is what lets a single
    phonon calculation serve the heat capacity and INS benchmarks.

    Parameters
    ----------
    model_name
        Name of the model whose outputs should be read.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of framework name to its phonon summary.
    """
    model_dir = OUTPUT_PATH / model_name
    summaries: dict[str, dict[str, Any]] = {}
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return summaries

    for path in sorted(model_dir.glob("*_phonon_summary.json")):
        try:
            with open(path, encoding="utf8") as handle:
                summaries[path.name.removesuffix("_phonon_summary.json")] = json.load(
                    handle
                )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to load {path}: {exc}")
    return summaries


def mean_imaginary_deviation(model_name: str) -> float | None:
    """
    Get a model's mean absolute imaginary-mode percentage.

    Every framework with a completed phonon calculation contributes, not only
    those carrying a reference for a scored quantity.

    Parameters
    ----------
    model_name
        Name of the model whose outputs should be read.

    Returns
    -------
    float | None
        Mean absolute deviation from zero imaginary modes, or ``None`` when
        the model produced no usable output.
    """
    values = [
        summary["percent_imaginary_modes"]
        for summary in load_phonon_summaries(model_name).values()
        if summary.get("percent_imaginary_modes") is not None
    ]
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return None
    return float(
        np.mean([abs(value - PERCENT_IMAGINARY_REFERENCE) for value in finite])
    )


def load_heat_capacity_reference() -> dict[str, dict[str, Any]]:
    """
    Load the experimental heat-capacity reference values.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of framework name to its reference record, which carries the
        value in J/g/K, the temperature, and the source DOI.
    """
    path = REFERENCE_PATH / HEAT_CAPACITY_REFERENCE_NAME
    if not path.exists():
        print(f"Reference data not found: {path}. Run the calculations first.")
        return {}
    with open(path, encoding="utf8") as handle:
        return json.load(handle)["data"]


# Equation-of-state scan used by the bulk modulus benchmark. Kept here rather
# than in that benchmark's calc module so it can be imported without pulling
# in the model registry. MOFs are compliant, so the strain range is modest:
# larger strains risk crossing a pressure-induced structural transition.
VOLUME_RANGE = 0.03
N_VOLUMES = 7


def volume_scales() -> np.ndarray:
    """
    Get the volume scale factors used for the equation-of-state scan.

    Returns
    -------
    np.ndarray
        Multiplicative factors on the relaxed cell volume, centred on 1;
        by default seven points spanning 0.97 to 1.03 in steps of 0.01.
    """
    return np.linspace(1.0 - VOLUME_RANGE, 1.0 + VOLUME_RANGE, N_VOLUMES)


def unconverged_eos_points(record: dict[str, Any]) -> list[float]:
    """
    Get the volume scale factors whose fixed-cell relaxation did not converge.

    Points that exhaust the inner step limit before reaching the force
    tolerance still enter the equation-of-state fit, so they are reported
    rather than dropped. Records written before this field existed are
    treated as fully converged.

    Parameters
    ----------
    record
        Contents of a ``<mof>_eos.json`` file.

    Returns
    -------
    list[float]
        Volume scale factors that hit the step limit, in scan order.
    """
    flags = record.get("inner_converged")
    if not flags:
        return []
    # Use the grid the record was written with, which need not be the current
    # one if VOLUME_RANGE or N_VOLUMES has changed since the scan was run.
    n_volumes = int(record.get("n_volumes", N_VOLUMES))
    half_width = float(record.get("volume_range", VOLUME_RANGE))
    if len(flags) != n_volumes:
        return []
    scales = np.linspace(1.0 - half_width, 1.0 + half_width, n_volumes)
    return [float(s) for s, ok in zip(scales, flags, strict=True) if not ok]


def load_bulk_modulus_reference() -> dict[str, dict[str, Any]]:
    """
    Load the experimental bulk-modulus reference values.

    Records carrying ``excluded: true`` are returned as-is; it is the caller's
    responsibility to drop them from any scored set, and each carries an
    ``exclusion_reason``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of framework name to its reference record, which holds the
        value in GPa and the source DOI.
    """
    path = REFERENCE_PATH / BULK_MODULUS_REFERENCE_NAME
    if not path.exists():
        print(f"Reference data not found: {path}. Run the calculations first.")
        return {}
    with open(path, encoding="utf8") as handle:
        return json.load(handle)["data"]


def load_reference_structures() -> list[str]:
    """
    List the frameworks the benchmark data was packaged with.

    Lets the analysis step resolve the scored set without needing the
    structure files themselves.

    Returns
    -------
    list[str]
        Sorted framework names, or an empty list when the reference data has
        not been staged.
    """
    path = REFERENCE_PATH / HEAT_CAPACITY_REFERENCE_NAME
    if not path.exists():
        return []
    with open(path, encoding="utf8") as handle:
        return sorted(json.load(handle).get("structures", []))


def load_ins_reference() -> dict[str, dict[str, Any]]:
    """
    Load the digitised reference INS spectra.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of framework name to reference type (``"experiment"`` or
        ``"DFT"``) to a record holding ``energy_meV`` and ``intensity``.
    """
    path = REFERENCE_PATH / INS_REFERENCE_NAME
    if not path.exists():
        print(f"Reference data not found: {path}. Run the calculations first.")
        return {}
    with gzip.open(path, "rt", encoding="utf8") as handle:
        return json.load(handle)["data"]
