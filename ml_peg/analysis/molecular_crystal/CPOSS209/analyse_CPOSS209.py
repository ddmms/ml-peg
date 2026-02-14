"""Analyse CPOSS209 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "molecular_crystal" / "CPOSS209" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_crystal" / "CPOSS209"


METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


def get_info() -> dict[str, list[str]]:
    """
    Get CPOSS209 system names, polymorphs, and file names.

    Returns
    -------
    dict[str, list[str]]
        Dictationary of info returned from first non-empty model directory.
    """
    info = {
        "systems": [],  # Directory names
        "polymorphs": [],  # Three-letter code + polymorph number
    }
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            for system_dir in sorted(model_dir.iterdir()):
                if system_dir.is_dir():
                    system_name = system_dir.name
                    info["systems"].append(system_name)

                    # Get crystal files (polymorphs) for this system
                    system_dir = model_dir / system_name
                    crystal_files = sorted(system_dir.glob("crystal*.xyz"))
                    # Get shortened name for each crystal file
                    for crystal_file in crystal_files:
                        crystal = read(system_dir / crystal_file, 0)
                        short_name = crystal.info["polymorph_name"]
                        info["polymorphs"].append(short_name)

            # Break after processing first model to avoid duplicates
            if info["systems"] and info["polymorphs"]:
                return info
    return info


INFO = get_info()


@pytest.fixture
def systems() -> list[str]:
    """
    Get list of CPOSS209 system names as a fixture.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    return INFO["systems"]


@pytest.fixture
def lattice_energies_raw(
    systems: list[str],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Calculate absolute and relative lattice energies for CPOSS209 benchmark systems.

    Parameters
    ----------
    systems
        List of CPOSS209 system names to analyze.

    Returns
    -------
    tuple[dict[str, list[float]], dict[str, list[float]]]
        A tuple containing:
        - First dict: Absolute lattice energies in kJ/mol for each model and reference.
          Keys are model names and "ref", values are lists of lattice energies.
        - Second dict: Relative lattice energies in kJ/mol (relative to minimum energy
          polymorph within each system). Same structure as first dict.

    Notes
    -----
    - Lattice energy = (crystal_energy / num_molecules) - min(molecule_energies)
    - Energies are converted from eV to kJ/mol
    - Reference energies are stored from crystal.info["ref"]
    - Structure files are written to OUT_PATH for each model and system
    """
    # Initialize result dictionaries: absolute and relative lattice energies
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_relative = {"ref": []} | {mlip: [] for mlip in MODELS}

    # Flag to ensure reference data is stored only once (same for all models)
    ref_stored = False

    # Loop through each model to compute predictions
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        # Skip if model directory doesn't exist
        if not model_dir.exists():
            continue

        # Process each system in the benchmark
        for system in systems:
            system_dir = model_dir / system
            # Find all crystal polymorph structure files
            crystal_files = sorted(system_dir.glob("crystal*.xyz"))

            # Find all gas-phase molecule structure files
            molecule_files = sorted(system_dir.glob("gas*.xyz"))

            # Process all gas-phase molecule structures to find minimum energy
            molecule_energies = []
            for molecule_file in molecule_files:
                # Read molecule structure
                molecule = read(system_dir / molecule_file, 0)

                # Write molecule files to output directory for reference
                out_dir = OUT_PATH / model_name / system
                out_dir.mkdir(parents=True, exist_ok=True)
                write(out_dir / molecule_file.name, molecule)

                # Extract energy from structure
                molecule_energy = molecule.get_potential_energy()
                molecule_energies.append(molecule_energy)

            # Use lowest molecule energy as reference for lattice energy calculation
            lowest_molecule_energy = min(molecule_energies)

            # Calculate lattice energies for all polymorphs (only read files once)
            lattice_energies_list = []
            reference_lattice_energies = []

            for crystal_file in crystal_files:
                # Read crystal structure once
                crystal = read(system_dir / crystal_file, 0)

                # Write crystal files to output directory for reference
                out_dir = OUT_PATH / model_name / system
                out_dir.mkdir(parents=True, exist_ok=True)
                write(out_dir / crystal_file.name, crystal)

                # Get crystal energy and number of molecules per unit cell
                crystal_energy = crystal.get_potential_energy()
                num_molecules = crystal.info["num_molecules"]

                # Calculate lattice energy
                # E_lattice = (E_crystal / n_molecules) - E_molecule
                lattice_energy = (
                    crystal_energy / num_molecules
                ) - lowest_molecule_energy

                # Convert from eV to kJ/mol
                lattice_energy_kj = lattice_energy * EV_TO_KJ_PER_MOL

                # Store for absolute energies
                results[model_name].append(lattice_energy_kj)
                lattice_energies_list.append(lattice_energy_kj)

                # Store reference data only once (same for all models)
                if not ref_stored:
                    ref_energy = crystal.info["ref"]
                    results["ref"].append(ref_energy)
                    reference_lattice_energies.append(ref_energy)

            # Find most stable polymorph (minimum lattice energy) for this model
            min_mlip_lattice_energy = min(lattice_energies_list)

            # Calculate relative energies: E_rel = E_polymorph - E_most_stable
            for le in lattice_energies_list:
                results_relative[model_name].append(le - min_mlip_lattice_energy)

            # Process reference relative energies only once
            if not ref_stored:
                # Find most stable reference polymorph
                min_ref_lattice_energy = min(reference_lattice_energies)

                # Calculate relative energies for reference data
                for rle in reference_lattice_energies:
                    results_relative["ref"].append(rle - min_ref_lattice_energy)

        # Mark reference data as stored after processing first model
        ref_stored = True

    return results, results_relative


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_absolute_lattice_energies.json",
    title="CPOSS209 Absolute Lattice Energies (All Polymorphs)",
    x_label="Predicted lattice energy / kJ/mol",
    y_label="Reference lattice energy / kJ/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def absolute_lattice_energies(
    lattice_energies_raw: tuple[dict[str, list[float]], dict[str, list[float]]],
) -> dict[str, list[float]]:
    """
    Get absolute lattice energies for all crystal polymorphs.

    Returns absolute lattice energies for all crystal structures,
    including all polymorphs for each system.

    Parameters
    ----------
    lattice_energies_raw
        Absolute and relative lattice energies for all systems.

    Returns
    -------
    dict
        Dictionary of absolute lattice energies with "ref" and model names as keys.
        Each entry contains lattice energy values for all crystal polymorphs.
    """
    # Return absolute lattice energies (index 0), which includes all crystal polymorphs
    return lattice_energies_raw[0]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_relative_lattice_energies.json",
    title="CPOSS209 Relative Lattice Energies (All Polymorphs)",
    x_label="Predicted relative lattice energy / kJ/mol",
    y_label="Reference relative lattice energy / kJ/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def relative_lattice_energies(
    lattice_energies_raw: tuple[dict[str, list[float]], dict[str, list[float]]],
) -> dict[str, list[float]]:
    """
    Get absolute lattice energies for all crystal polymorphs.

    Returns absolute lattice energies for all crystal structures,
    including all polymorphs for each system.

    Parameters
    ----------
    lattice_energies_raw
        Absolute and relative lattice energies for all systems.

    Returns
    -------
    dict
        Dictionary of absolute lattice energies with "ref" and model names as keys.
        Each entry contains lattice energy values for all crystal polymorphs.
    """
    # Return absolute lattice energies (index 0), which includes all crystal polymorphs
    return lattice_energies_raw[1]


@pytest.fixture
def cposs209_errors(
    absolute_lattice_energies: dict[str, list[float]],
    relative_lattice_energies: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Get mean absolute error for absolute and relative lattice energies.

    Parameters
    ----------
    absolute_lattice_energies
        Dictionary of absolute reference and predicted lattice energies.
    relative_lattice_energies
        Dictionary of relative reference and predicted lattice energies.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        Tuple of (absolute_errors, relative_errors) dictionaries for all models.
    """
    results_absolute = {}
    for model_name in MODELS:
        if absolute_lattice_energies[model_name]:
            results_absolute[model_name] = mae(
                absolute_lattice_energies["ref"], absolute_lattice_energies[model_name]
            )

        else:
            results_absolute[model_name] = None

    results_relative = {}
    for model_name in MODELS:
        if relative_lattice_energies[model_name]:
            results_relative[model_name] = mae(
                relative_lattice_energies["ref"], relative_lattice_energies[model_name]
            )

        else:
            results_relative[model_name] = None

    return results_absolute, results_relative


@pytest.fixture
@build_table(
    filename=OUT_PATH / "cposs209_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    cposs209_errors: tuple[dict[str, float], dict[str, float]],
) -> dict[str, dict]:
    """
    Get all CPOSS209 metrics.

    Parameters
    ----------
    cposs209_errors
        Tuple of (absolute_errors, relative_errors) mean absolute errors for all
        systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    absolute_errors, relative_errors = cposs209_errors
    return {
        "Absolute MAE": absolute_errors,
        "Relative MAE": relative_errors,
    }


def test_cposs209(metrics: dict[str, dict]) -> None:
    """
    Run CPOSS209 test.

    Parameters
    ----------
    metrics
        All CPOSS209 metrics.
    """
    return
