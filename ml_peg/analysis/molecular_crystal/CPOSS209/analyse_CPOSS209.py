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
EV_TO_KCAL_PER_MOL = units.mol / units.kcal
KJ_PER_MOL_TO_KCAL_PER_MOL = units.kJ / units.kcal


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
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    """
    Calculate absolute and relative lattice energies for CPOSS209 benchmark systems.

    Parameters
    ----------
    systems
        List of CPOSS209 system names to analyze.

    Returns
    -------
    tuple
        A 10-dict tuple containing (in order):
        1) Absolute lattice energies in kcal/mol for each model and reference.
        2) Relative lattice energies in kcal/mol for each model and reference.
        3-4) Absolute and relative energies for small rigid molecules.
        5-6) Absolute and relative energies for the carbamazepine family.
        7-8) Absolute and relative energies for the fenamate family.
        9-10) Absolute and relative energies for small drug molecules.

    Notes
    -----
    - Lattice energy = (crystal_energy / num_molecules) - min(molecule_energies)
    - Energies are converted from eV to kcal/mol
    - Reference energies are stored from crystal.info["ref"]
    - Structure files are written to OUT_PATH for each model and system
    """
    # Initialize result dictionaries: absolute and relative lattice energies
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_relative = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_absolute_small_rigid_molecules = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_relative_small_rigid_molecules = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_absolute_carbamazepine_family = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_relative_carbamazepine_family = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_absolute_fenamate_family = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_relative_fenamate_family = {"ref": []} | {mlip: [] for mlip in MODELS}
    results_absolute_small_drug_molecule_family = {"ref": []} | {
        mlip: [] for mlip in MODELS
    }
    results_relative_small_drug_molecule_family = {"ref": []} | {
        mlip: [] for mlip in MODELS
    }

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
            lattice_energies_families = []
            reference_lattice_energies_families = []

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

                # Convert from eV to kcal/mol
                lattice_energy_kcal = lattice_energy * EV_TO_KCAL_PER_MOL

                # Store for absolute energies
                results[model_name].append(lattice_energy_kcal)
                lattice_energies_list.append(lattice_energy_kcal)

                # Track family for later per-family relative calculations
                family = crystal.info["molecular_family"]
                ref_energy = crystal.info["ref"] * KJ_PER_MOL_TO_KCAL_PER_MOL
                lattice_energies_families.append(family)
                reference_lattice_energies_families.append(family)

                # Store by-family absolute energies (reference once)
                if family == "Small_rigid_molecules":
                    results_absolute_small_rigid_molecules[model_name].append(
                        lattice_energy_kcal
                    )
                    if not ref_stored:
                        results_absolute_small_rigid_molecules["ref"].append(ref_energy)

                elif family == "Carbamazepine_family":
                    results_absolute_carbamazepine_family[model_name].append(
                        lattice_energy_kcal
                    )
                    if not ref_stored:
                        results_absolute_carbamazepine_family["ref"].append(ref_energy)

                elif family == "Fenamate_family":
                    results_absolute_fenamate_family[model_name].append(
                        lattice_energy_kcal
                    )
                    if not ref_stored:
                        results_absolute_fenamate_family["ref"].append(ref_energy)

                elif family == "Small_drug_molecules":
                    results_absolute_small_drug_molecule_family[model_name].append(
                        lattice_energy_kcal
                    )
                    if not ref_stored:
                        results_absolute_small_drug_molecule_family["ref"].append(
                            ref_energy
                        )

                # Store reference data only once (same for all models)
                if not ref_stored:
                    results["ref"].append(ref_energy)
                    reference_lattice_energies.append(ref_energy)

            # Find most stable polymorph (minimum lattice energy) for this model
            min_mlip_lattice_energy = min(lattice_energies_list)

            # Calculate relative energies: E_rel = E_polymorph - E_most_stable
            for le, fam in zip(
                lattice_energies_list, lattice_energies_families, strict=False
            ):
                rel_val = le - min_mlip_lattice_energy
                results_relative[model_name].append(rel_val)
                if fam == "Small_rigid_molecules":
                    results_relative_small_rigid_molecules[model_name].append(rel_val)
                elif fam == "Carbamazepine_family":
                    results_relative_carbamazepine_family[model_name].append(rel_val)
                elif fam == "Fenamate_family":
                    results_relative_fenamate_family[model_name].append(rel_val)
                elif fam == "Small_drug_molecules":
                    results_relative_small_drug_molecule_family[model_name].append(
                        rel_val
                    )

            # Process reference relative energies only once
            if not ref_stored:
                # Find most stable reference polymorph
                min_ref_lattice_energy = min(reference_lattice_energies)

                # Calculate relative energies for reference data
                for rle, fam in zip(
                    reference_lattice_energies,
                    reference_lattice_energies_families,
                    strict=False,
                ):
                    rel_ref_val = rle - min_ref_lattice_energy
                    results_relative["ref"].append(rel_ref_val)
                    if fam == "Small_rigid_molecules":
                        results_relative_small_rigid_molecules["ref"].append(
                            rel_ref_val
                        )
                    elif fam == "Carbamazepine_family":
                        results_relative_carbamazepine_family["ref"].append(rel_ref_val)
                    elif fam == "Fenamate_family":
                        results_relative_fenamate_family["ref"].append(rel_ref_val)
                    elif fam == "Small_drug_molecules":
                        results_relative_small_drug_molecule_family["ref"].append(
                            rel_ref_val
                        )

        # Mark reference data as stored after processing first model
        ref_stored = True

    return (
        results,
        results_relative,
        results_absolute_small_rigid_molecules,
        results_relative_small_rigid_molecules,
        results_absolute_carbamazepine_family,
        results_relative_carbamazepine_family,
        results_absolute_fenamate_family,
        results_relative_fenamate_family,
        results_absolute_small_drug_molecule_family,
        results_relative_small_drug_molecule_family,
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_absolute_lattice_energies.json",
    title="CPOSS209 Absolute Lattice Energies (All Polymorphs)",
    x_label="Predicted lattice energy / kcal/mol",
    y_label="Reference lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def absolute_lattice_energies(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
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
    x_label="Predicted relative lattice energy / kcal/mol",
    y_label="Reference relative lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def relative_lattice_energies(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
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
@plot_parity(
    filename=OUT_PATH / "figure_absolute_lattice_energies_small_rigid_molecules.json",
    title="CPOSS209 Absolute Lattice Energies for Small Rigid Molecules",
    x_label="Predicted lattice energy / kcal/mol",
    y_label="Reference lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def absolute_lattice_energies_small_rigid_molecules(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
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
    # Return absolute lattice energies by family (index 2).
    # Includes all crystal polymorphs
    return lattice_energies_raw[2]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_relative_lattice_energies_small_rigid_molecules.json",
    title="CPOSS209 Relative Lattice Energies for Small Rigid Molecules",
    x_label="Predicted relative lattice energy / kcal/mol",
    y_label="Reference relative lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def relative_lattice_energies_small_rigid_molecules(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
) -> dict[str, list[float]]:
    """
    Get relative lattice energies for all crystal polymorphs.

    Returns relative lattice energies for all crystal structures,
    including all polymorphs for each system.

    Parameters
    ----------
    lattice_energies_raw
        Absolute and relative lattice energies for all systems.

    Returns
    -------
    dict
        Dictionary of relative lattice energies with "ref" and model names as keys.
        Each entry contains lattice energy values for all crystal polymorphs.
    """
    # Return relative lattice energies by family (index 3).
    # Includes all crystal polymorphs
    return lattice_energies_raw[3]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_absolute_lattice_energies_carbamazepine_family.json",
    title="CPOSS209 Absolute Lattice Energies for Carbamazepine Family",
    x_label="Predicted lattice energy / kcal/mol",
    y_label="Reference lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def absolute_lattice_energies_carbamazepine_family(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
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
    # Return absolute lattice energies by family (index 4).
    # Includes all crystal polymorphs
    return lattice_energies_raw[4]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_relative_lattice_energies_carbamazepine_family.json",
    title="CPOSS209 Relative Lattice Energies for Carbamazepine Family",
    x_label="Predicted relative lattice energy / kcal/mol",
    y_label="Reference relative lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def relative_lattice_energies_carbamazepine_family(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
) -> dict[str, list[float]]:
    """
    Get relative lattice energies for all crystal polymorphs.

    Returns relative lattice energies for all crystal structures,
    including all polymorphs for each system.

    Parameters
    ----------
    lattice_energies_raw
        Absolute and relative lattice energies for all systems.

    Returns
    -------
    dict
        Dictionary of relative lattice energies with "ref" and model names as keys.
        Each entry contains lattice energy values for all crystal polymorphs.
    """
    # Return relative lattice energies by family (index 5).
    # Includes all crystal polymorphs
    return lattice_energies_raw[5]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_absolute_lattice_energies_fenamate_family.json",
    title="CPOSS209 Absolute Lattice Energies for Fenamate Family",
    x_label="Predicted lattice energy / kcal/mol",
    y_label="Reference lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def absolute_lattice_energies_fenamate_family(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
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
    # Return absolute lattice energies by family (index 6).
    # Includes all crystal polymorphs
    return lattice_energies_raw[6]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_relative_lattice_energies_fenamate_family.json",
    title="CPOSS209 Relative Lattice Energies for Fenamate Family",
    x_label="Predicted relative lattice energy / kcal/mol",
    y_label="Reference relative lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def relative_lattice_energies_fenamate_family(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
) -> dict[str, list[float]]:
    """
    Get relative lattice energies for all crystal polymorphs.

    Returns relative lattice energies for all crystal structures,
    including all polymorphs for each system.

    Parameters
    ----------
    lattice_energies_raw
        Absolute and relative lattice energies for all systems.

    Returns
    -------
    dict
        Dictionary of relative lattice energies with "ref" and model names as keys.
        Each entry contains lattice energy values for all crystal polymorphs.
    """
    # Return relative lattice energies by family (index 7).
    # Includes all crystal polymorphs
    return lattice_energies_raw[7]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH
    / "figure_absolute_lattice_energies_small_drug_molecule_family.json",
    title="CPOSS209 Absolute Lattice Energies for Small Drug Molecule Family",
    x_label="Predicted lattice energy / kcal/mol",
    y_label="Reference lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def absolute_lattice_energies_small_drug_molecule_family(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
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
    # Return absolute lattice energies by family (index 8).
    # Includes all crystal polymorphs
    return lattice_energies_raw[8]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH
    / "figure_relative_lattice_energies_small_drug_molecule_family.json",
    title="CPOSS209 Relative Lattice Energies for Small Drug Molecule Family",
    x_label="Predicted relative lattice energy / kcal/mol",
    y_label="Reference relative lattice energy / kcal/mol",
    hoverdata={
        "Crystal": INFO["polymorphs"],
    },
)
def relative_lattice_energies_small_drug_molecule_family(
    lattice_energies_raw: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ],
) -> dict[str, list[float]]:
    """
    Get relative lattice energies for all crystal polymorphs.

    Returns relative lattice energies for all crystal structures,
    including all polymorphs for each system.

    Parameters
    ----------
    lattice_energies_raw
        Absolute and relative lattice energies for all systems.

    Returns
    -------
    dict
        Dictionary of relative lattice energies with "ref" and model names as keys.
        Each entry contains lattice energy values for all crystal polymorphs.
    """
    # Return relative lattice energies by family (index 9).
    # Includes all crystal polymorphs
    return lattice_energies_raw[9]


@pytest.fixture
def cposs209_errors(
    absolute_lattice_energies: dict[str, list[float]],
    relative_lattice_energies: dict[str, list[float]],
    absolute_lattice_energies_small_rigid_molecules: dict[str, list[float]],
    relative_lattice_energies_small_rigid_molecules: dict[str, list[float]],
    absolute_lattice_energies_carbamazepine_family: dict[str, list[float]],
    relative_lattice_energies_carbamazepine_family: dict[str, list[float]],
    absolute_lattice_energies_fenamate_family: dict[str, list[float]],
    relative_lattice_energies_fenamate_family: dict[str, list[float]],
    absolute_lattice_energies_small_drug_molecule_family: dict[str, list[float]],
    relative_lattice_energies_small_drug_molecule_family: dict[str, list[float]],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """
    Get mean absolute error for absolute and relative lattice energies.

    Parameters
    ----------
    absolute_lattice_energies
        Dictionary of absolute reference and predicted lattice energies.
    relative_lattice_energies
        Dictionary of relative reference and predicted lattice energies.
    absolute_lattice_energies_small_rigid_molecules
        Dictionary of absolute reference and predicted lattice energies for the
        small rigid molecule family.
    relative_lattice_energies_small_rigid_molecules
        Dictionary of relative reference and predicted lattice energies for the
        small rigid molecule family.
    absolute_lattice_energies_carbamazepine_family
        Dictionary of absolute reference and predicted lattice energies for the
        carbamazepine family.
    relative_lattice_energies_carbamazepine_family
        Dictionary of relative reference and predicted lattice energies for the
        carbamazepine family.
    absolute_lattice_energies_fenamate_family
        Dictionary of absolute reference and predicted lattice energies for the
        fenamate family.
    relative_lattice_energies_fenamate_family
        Dictionary of relative reference and predicted lattice energies for the
        fenamate family.
    absolute_lattice_energies_small_drug_molecule_family
        Dictionary of absolute reference and predicted lattice energies for the
        small drug molecule family.
    relative_lattice_energies_small_drug_molecule_family
        Dictionary of relative reference and predicted lattice energies for the
        small drug molecule family.

    Returns
    -------
    tuple
        A 10-dict tuple containing (in order):
        1) Absolute MAE (all polymorphs)
        2) Relative MAE (all polymorphs)
        3-4) Absolute and relative MAE for small rigid molecules
        5-6) Absolute and relative MAE for the carbamazepine family
        7-8) Absolute and relative MAE for the fenamate family
        9-10) Absolute and relative MAE for small drug molecules.
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

    results_absolute_small_rigid_molecules = {}
    for model_name in MODELS:
        if absolute_lattice_energies_small_rigid_molecules[model_name]:
            results_absolute_small_rigid_molecules[model_name] = mae(
                absolute_lattice_energies_small_rigid_molecules["ref"],
                absolute_lattice_energies_small_rigid_molecules[model_name],
            )

        else:
            results_absolute_small_rigid_molecules[model_name] = None

    results_relative_small_rigid_molecules = {}
    for model_name in MODELS:
        if relative_lattice_energies_small_rigid_molecules[model_name]:
            results_relative_small_rigid_molecules[model_name] = mae(
                relative_lattice_energies_small_rigid_molecules["ref"],
                relative_lattice_energies_small_rigid_molecules[model_name],
            )

        else:
            results_relative_small_rigid_molecules[model_name] = None

    results_absolute_carbamazepine_family = {}
    for model_name in MODELS:
        if absolute_lattice_energies_carbamazepine_family[model_name]:
            results_absolute_carbamazepine_family[model_name] = mae(
                absolute_lattice_energies_carbamazepine_family["ref"],
                absolute_lattice_energies_carbamazepine_family[model_name],
            )

        else:
            results_absolute_carbamazepine_family[model_name] = None

    results_relative_carbamazepine_family = {}
    for model_name in MODELS:
        if relative_lattice_energies_carbamazepine_family[model_name]:
            results_relative_carbamazepine_family[model_name] = mae(
                relative_lattice_energies_carbamazepine_family["ref"],
                relative_lattice_energies_carbamazepine_family[model_name],
            )

        else:
            results_relative_carbamazepine_family[model_name] = None

    results_absolute_fenamate_family = {}
    for model_name in MODELS:
        if absolute_lattice_energies_fenamate_family[model_name]:
            results_absolute_fenamate_family[model_name] = mae(
                absolute_lattice_energies_fenamate_family["ref"],
                absolute_lattice_energies_fenamate_family[model_name],
            )

        else:
            results_absolute_fenamate_family[model_name] = None

    results_relative_fenamate_family = {}
    for model_name in MODELS:
        if relative_lattice_energies_fenamate_family[model_name]:
            results_relative_fenamate_family[model_name] = mae(
                relative_lattice_energies_fenamate_family["ref"],
                relative_lattice_energies_fenamate_family[model_name],
            )

        else:
            results_relative_fenamate_family[model_name] = None

    results_absolute_small_drug_molecule_family = {}
    for model_name in MODELS:
        if absolute_lattice_energies_small_drug_molecule_family[model_name]:
            results_absolute_small_drug_molecule_family[model_name] = mae(
                absolute_lattice_energies_small_drug_molecule_family["ref"],
                absolute_lattice_energies_small_drug_molecule_family[model_name],
            )

        else:
            results_absolute_small_drug_molecule_family[model_name] = None

    results_relative_small_drug_molecule_family = {}
    for model_name in MODELS:
        if relative_lattice_energies_small_drug_molecule_family[model_name]:
            results_relative_small_drug_molecule_family[model_name] = mae(
                relative_lattice_energies_small_drug_molecule_family["ref"],
                relative_lattice_energies_small_drug_molecule_family[model_name],
            )

        else:
            results_relative_small_drug_molecule_family[model_name] = None

    return (
        results_absolute,
        results_relative,
        results_absolute_small_rigid_molecules,
        results_relative_small_rigid_molecules,
        results_absolute_carbamazepine_family,
        results_relative_carbamazepine_family,
        results_absolute_fenamate_family,
        results_relative_fenamate_family,
        results_absolute_small_drug_molecule_family,
        results_relative_small_drug_molecule_family,
    )


@pytest.fixture
@build_table(
    filename=OUT_PATH / "cposs209_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    cposs209_errors: tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ],
) -> dict[str, dict]:
    """
    Get all CPOSS209 metrics.

    Parameters
    ----------
    cposs209_errors
        A 10-dict tuple of mean absolute errors matching the order produced by
        `cposs209_errors` (global absolute/relative, then per-family absolute/relative).

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    (
        absolute_errors,
        relative_errors,
        small_rigid_family_absolute_errors,
        small_rigid_family_relative_errors,
        carbamazepine_family_absolute_errors,
        carbamazepine_family_relative_errors,
        fenamate_family_absolute_errors,
        fenamate_family_relative_errors,
        small_drug_molecule_family_absolute_errors,
        small_drug_molecule_family_relative_errors,
    ) = cposs209_errors
    return {
        "Absolute MAE": absolute_errors,
        "Relative MAE": relative_errors,
        "Absolute MAE small rigid molecules": small_rigid_family_absolute_errors,
        "Relative MAE small rigid molecules": small_rigid_family_relative_errors,
        "Absolute MAE carbamazepine family": carbamazepine_family_absolute_errors,
        "Relative MAE carbamazepine family": carbamazepine_family_relative_errors,
        "Absolute MAE fenamate family": fenamate_family_absolute_errors,
        "Relative MAE fenamate family": fenamate_family_relative_errors,
        "Absolute MAE small drug molecules": small_drug_molecule_family_absolute_errors,
        "Relative MAE small drug molecules": small_drug_molecule_family_relative_errors,
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
