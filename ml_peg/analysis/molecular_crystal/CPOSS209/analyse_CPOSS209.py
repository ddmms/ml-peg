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

# Set data path
DATA_PATH = CALC_PATH


def get_system_names() -> list[str]:
    """
    Get list of CPOSS209 system names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    system_names = []
    # Only get system names from first available model to avoid duplicates
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            for system_dir in sorted(model_dir.iterdir()):
                if system_dir.is_dir():
                    system_names.append(system_dir.name)
            # Break after processing first model to avoid duplicates
            break
    return system_names


def get_system_names_with_polymorphs() -> list[str]:
    """
    Get system names repeated for each crystal polymorph.
    
    Returns a list where each system name appears once for each of its crystal
    polymorphs, matching the data structure in lattice_energies.

    Returns
    -------
    list[str]
        List of system names repeated for each polymorph.
    """
    system_names_expanded = []
    
    # Find the first available model to get the structure
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            for system_dir in sorted(model_dir.iterdir()):
                if system_dir.is_dir():
                    system_name = system_dir.name
                    # Count crystal files (polymorphs) for this system
                    crystal_files = sorted([
                        f for f in (DATA_PATH / model_name / system_name).iterdir() 
                        if f.name.startswith("crystal") and f.name.endswith(".xyz")
                    ])
                    # Add system name once for each crystal polymorph
                    for _ in crystal_files:
                        system_names_expanded.append(system_name)
            # Only process one model to get the structure
            break
    
    return system_names_expanded


def get_crystal_file_names() -> list[str]:
    """
    Get shortened crystal names for each polymorph.
    
    Returns a list of shortened crystal names (three-letter code + polymorph number)
    in the same order as the lattice energies are processed.

    Returns
    -------
    list[str]
        List of shortened crystal names for each polymorph (e.g., "ACR1", "ACR2").
    """
    crystal_file_names = []
    
    # Find the first available model to get the structure
    for model_name in MODELS:
        model_dir = DATA_PATH / model_name
        if model_dir.exists():
            for system_dir in sorted(model_dir.iterdir()):
                if system_dir.is_dir():
                    system_name = system_dir.name
                    # Get crystal files (polymorphs) for this system
                    crystal_files = sorted([
                        f for f in (DATA_PATH / model_name / system_name).iterdir() 
                        if f.name.startswith("crystal") and f.name.endswith(".xyz")
                    ])
                    # Get shortened name for each crystal file
                    for crystal_file in crystal_files:
                        crystal = read(DATA_PATH / model_name / system_name / crystal_file, 0)
                        short_name = crystal.info["polymorph_name"]
                        crystal_file_names.append(short_name)
            # Only process one model to get the structure
            break
    
    return crystal_file_names


@pytest.fixture
def systems() -> list[str]:
    """
    Get list of CPOSS209 system names as a fixture.
    
    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    return get_system_names()


@pytest.fixture
def lattice_energies_raw(systems):
    """
    Calculate absolute and relative lattice energies for CPOSS209 benchmark systems.
    
    Parameters
    ----------
    systems : list[str]
        List of CPOSS209 system names to analyze.
    
    Returns
    -------
    tuple[dict[str, list[float]], dict[str, list[float]], list[str]]
        A tuple containing:
        - First dict: Absolute lattice energies in kJ/mol for each model and reference.
          Keys are model names and "ref", values are lists of lattice energies.
        - Second dict: Relative lattice energies in kJ/mol (relative to minimum energy
          polymorph within each system). Same structure as first dict.
        - Third list: Crystal names (shortened format: three-letter code + polymorph number).
    
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
    
    # List to store crystal names for hover data
    crystal_names = []

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
            # Find all crystal polymorph structure files
            crystal_files = [file for file in (DATA_PATH / model_name / system).iterdir() if file.name.startswith("crystal") and file.name.endswith(".xyz")]
            crystal_files = sorted(crystal_files)

            if not crystal_files:
                continue

            # Find all gas-phase molecule structure files
            molecule_files = [file for file in (DATA_PATH / model_name / system).iterdir() if file.name.startswith("gas") and file.name.endswith(".xyz")]
            molecule_files = sorted(molecule_files)

            # Skip system if no gas-phase molecules found
            if not molecule_files:
                continue

            # Process all gas-phase molecule structures to find minimum energy
            molecule_energies = []
            for molecule_file in molecule_files:
                # Read molecule structure
                molecule = read(DATA_PATH / model_name / system / molecule_file, 0)

                # Write molecule files to output directory for reference
                dir = OUT_PATH / model_name / system
                dir.mkdir(parents=True, exist_ok=True)
                write(dir / molecule_file.name, molecule)

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
                crystal = read(DATA_PATH / model_name / system / crystal_file, 0)
                
                short_name = crystal.info["polymorph_name"]
                
                # Store crystal name only once (same for all models)
                if not ref_stored:
                    crystal_names.append(short_name)
                
                # Write crystal files to output directory for reference
                dir = OUT_PATH / model_name / system
                dir.mkdir(parents=True, exist_ok=True)
                write(dir / crystal_file.name, crystal)
                
                # Get crystal energy and number of molecules per unit cell
                crystal_energy = crystal.get_potential_energy()
                num_molecules = crystal.info["num_molecules"]

                # Calculate lattice energy: E_lattice = (E_crystal / n_molecules) - E_molecule
                lattice_energy = (crystal_energy / num_molecules) - lowest_molecule_energy

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

    return results, results_relative, crystal_names


@pytest.fixture
def lattice_energies(lattice_energies_raw):
    """
    Get absolute lattice energies for all crystal polymorphs.
    
    Returns absolute lattice energies for all crystal structures,
    including all polymorphs for each system.
    
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
    filename=OUT_PATH / "figure_lattice_energies.json",
    title="CPOSS209 Absolute Lattice Energies (All Polymorphs)",
    x_label="Predicted lattice energy / kJ/mol",
    y_label="Reference lattice energy / kJ/mol",
    hoverdata={
        "Crystal": get_crystal_file_names(),
    },
)
def lattice_energies_plot(lattice_energies):
    """
    Plot absolute lattice energies with crystal names in hover data.
    
    Returns
    -------
    dict
        Dictionary of absolute lattice energies.
    """
    # Just return the energies dict for the plot
    return lattice_energies


@pytest.fixture
def cposs209_errors(lattice_energies_raw) -> tuple[dict[str, float], dict[str, float]]:
    """
    Get mean absolute error for absolute and relative lattice energies.

    Parameters
    ----------
    lattice_energies_raw
        Tuple of (absolute, relative, crystal_names) dictionaries of reference and predicted lattice energies.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        Tuple of (absolute_errors, relative_errors) dictionaries for all models.
    """
    absolute_lattice_energies, relative_lattice_energies, _ = lattice_energies_raw
    results = {}
    for model_name in MODELS:
        if absolute_lattice_energies[model_name]:
            results[model_name] = mae(
                absolute_lattice_energies["ref"], absolute_lattice_energies[model_name]
            )

        else:
            results[model_name] = None
    
    results_relative = {}
    for model_name in MODELS:
        if relative_lattice_energies[model_name]:
            results_relative[model_name] = mae(
                relative_lattice_energies["ref"], relative_lattice_energies[model_name]
            )

        else:
            results_relative[model_name] = None

    return results, results_relative


@pytest.fixture
@build_table(
    filename=OUT_PATH / "cposs209_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(cposs209_errors: tuple[dict[str, float], dict[str, float]]) -> dict[str, dict]:
    """
    Get all CPOSS209 metrics.

    Parameters
    ----------
    cposs209_errors
        Tuple of (absolute_errors, relative_errors) mean absolute errors for all systems.

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


def test_cposs209(metrics: dict[str, dict], lattice_energies_plot: dict) -> None:
    """
    Run CPOSS209 test.

    Parameters
    ----------
    metrics
        All CPOSS209 metrics.
    lattice_energies_plot
        Lattice energies plot data.
    """
    return
