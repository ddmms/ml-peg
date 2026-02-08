"""Analyse split vacancy benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
import pytest
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "split_vacancy" / "outputs"
CALC_PATH_PBESOL = CALC_PATH / "pbesol"  # oxides
CALC_PATH_PBE = CALC_PATH / "pbe"  # nitrides
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "split_vacancy"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# same setting as MatBench
# https://github.com/janosh/matbench-discovery/blob/93cc6907ac08b4adaa8391ccc4adf9c015c0dd61/matbench_discovery/structure/symmetry.py#L124
STRUCTURE_MATCHER = StructureMatcher(stol=1.0, scale=False)


def get_rmsd(atoms_1, atoms_2) -> float:
    """
    Calculate the RMSD between two ASE Atoms objects.

    Parameters
    ----------
    atoms_1, atoms_2
        ASE Atoms objects.

    Returns
    -------
    float
        Root mean square displacement.
    """
    rmsd, max_dist = STRUCTURE_MATCHER.get_rms_dist(
        Structure.from_ase_atoms(atoms_1), Structure.from_ase_atoms(atoms_2)
    )

    return rmsd


def get_hoverdata(functional_path) -> tuple[list, list, list]:
    """
    Get hover data.

    Returns
    -------
    tuple[list, list, list ]
        Tuple of Materials Project IDs, bulk formulae and vacant cations.
    """
    # TODO: RMSD could be good hoverdata?
    # TODO: add cell charge?
    mp_ids = []
    formulae = []
    vacant_cations = []

    model_dir = model_dir = functional_path / MODELS[0]
    for material_dir in tqdm(list(model_dir.iterdir())):
        split_dir_name = material_dir.stem.split("-")
        bulk_formula = split_dir_name[0]
        mp_id = f"mp-{split_dir_name[-1]}"

        cation_dirs = [
            p for p in material_dir.iterdir() if p.is_dir()
        ]  # skip pristine supercell.xyz files if present (not used)

        for cation_dir in cation_dirs:
            cation = cation_dir.stem

            mp_ids.append(mp_id)
            formulae.append(bulk_formula)
            vacant_cations.append(cation)

    return mp_ids, formulae, vacant_cations


MP_IDS_PBE, BULK_FORMULAE_PBE, VACANT_CATIONS_PBE = get_hoverdata(CALC_PATH_PBE)
MP_IDS_PBESOL, BULK_FORMULAE_PBESOL, VACANT_CATIONS_PBESOL = get_hoverdata(
    CALC_PATH_PBESOL
)


def build_results(
    functional_path,
) -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
    """
    Iterate through bulk-cation pairs calculating results.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, float]]
        Tuple of metrics.
    """
    # preference_energy_threshold = 0  # TODO: confirm

    print(f"Analysing {functional_path.stem} calculations.")

    result_formation_energy = {"ref": []} | {
        mlip: [] for mlip in MODELS
    }  # formation energy for every material-cation pair
    result_spearmans_coefficient = {
        mlip: [] for mlip in MODELS
    }  # spearmans coefficient for every material-cation pair
    result_rmsd = {
        mlip: [] for mlip in MODELS
    }  # RMSD error for every material-cation pair
    # TODO: investigate Kendall rank correlation
    result_rmsd = {mlip: [] for mlip in MODELS}

    ref_stored = False
    for model_name in tqdm(MODELS):
        model_dir = functional_path / model_name

        if not model_dir.exists():
            continue

        for material_dir in tqdm(list(model_dir.iterdir()), leave=False):
            cation_dirs = [
                p for p in material_dir.iterdir() if p.is_dir()
            ]  # skip pristine supercell.xyz files if present (not used)

            for cation_dir in tqdm(cation_dirs, leave=False):
                # cation = cation_dir.stem TODO: save structures for visualization

                nv_xyz_path = cation_dir / "normal_vacancy.xyz.gz"
                sv_xyz_path = cation_dir / "split_vacancy.xyz.gz"

                if not (nv_xyz_path.exists() and sv_xyz_path.exists()):
                    raise ValueError  # TODO: remove

                nv_atoms_list = read(nv_xyz_path, ":")
                sv_atoms_list = read(sv_xyz_path, ":")

                if not ref_stored:
                    ref_nv_energies = [
                        float(at.info["ref_energy"]) for at in nv_atoms_list
                    ]
                    ref_sv_energies = [
                        float(at.info["ref_energy"]) for at in sv_atoms_list
                    ]

                    ref_sv_formation_energy = min(ref_sv_energies) - min(
                        ref_nv_energies
                    )
                    # ref_sv_preferred = (
                    #     ref_sv_formation_energy < preference_energy_threshold
                    # ) # TODO: F1 score

                    result_formation_energy["ref"].append(ref_sv_formation_energy)

                    ref_cation_dir = (
                        functional_path / "ref" / material_dir.stem / cation_dir.stem
                    )
                    ref_nv_atoms_list = read(
                        ref_cation_dir / "normal_vacancy.xyz.gz", ":"
                    )
                    ref_sv_atoms_list = read(
                        ref_cation_dir / "split_vacancy.xyz.gz", ":"
                    )

                nv_energies = [float(at.info["relaxed_energy"]) for at in nv_atoms_list]
                sv_energies = [float(at.info["relaxed_energy"]) for at in sv_atoms_list]

                # calculate metrics
                sv_formation_energy = min(sv_energies) - min(nv_energies)
                # sv_preferred = sv_formation_energy < preference_energy_threshold

                spearmans_coefficient = spearmanr(
                    [float(at.info["initial_energy"]) for at in nv_atoms_list]
                    + [float(at.info["initial_energy"]) for at in sv_atoms_list],
                    ref_sv_energies + ref_nv_energies,
                ).statistic

                rmsd_list = []
                for ref_atoms, mlip_atoms in zip(
                    ref_nv_atoms_list, nv_atoms_list, strict=False
                ):
                    rmsd_list.append(get_rmsd(ref_atoms, mlip_atoms))
                for ref_atoms, mlip_atoms in zip(
                    ref_sv_atoms_list, sv_atoms_list, strict=False
                ):
                    rmsd_list.append(get_rmsd(ref_atoms, mlip_atoms))

                # add metrics to dicts
                result_formation_energy[model_name].append(sv_formation_energy)
                result_spearmans_coefficient[model_name].append(spearmans_coefficient)
                result_rmsd[model_name].extend(rmsd_list)

        ref_stored = False
    return result_formation_energy, result_spearmans_coefficient, result_rmsd


@pytest.fixture  # cache outputs
def build_results_pbesol():
    """
    Get PBEsol (oxide) results.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, float]]
        Tuple of metrics.
    """
    return build_results(CALC_PATH_PBESOL)


@pytest.fixture  # cache outputs
def build_results_pbe():
    """
    Get PBE (nitride) results.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, float]]
        Tuple of metrics.
    """
    return build_results(CALC_PATH_PBE)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_formation_energies_pbesol.json",
    title="Split vacancy (Oxides, PBEsol)",
    x_label="Predicted Split Vacancy Formation Energy / eV",
    y_label="DFT Split Vacancy Formation Energy / eV",
    hoverdata={
        "Materials Project ID": MP_IDS_PBESOL,
        "Formula": BULK_FORMULAE_PBESOL,
        "Vacant Cation": VACANT_CATIONS_PBESOL,
    },
)
def formation_energies_pbesol(build_results_pbesol) -> dict[str, list]:
    """
    Get DFT and predicted formation energies for oxides (PBEsol).

    Parameters
    ----------
    build_results
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, list]
        Dictionary of DFT and predicted formation energies.
    """
    result_formation_energies, _, _ = build_results_pbesol()
    return result_formation_energies


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_formation_energies_pbe.json",
    title="Split vacancy (Nitrides, PBE(+U))",
    x_label="Predicted Split Vacancy Formation Energy / eV",
    y_label="DFT Split Vacancy Formation Energy / eV",
    hoverdata={
        "Materials Project ID": MP_IDS_PBE,
        "Formula": BULK_FORMULAE_PBE,
        "Vacant Cation": VACANT_CATIONS_PBE,
    },
)
def formation_energies_pbe(build_results_pbe) -> dict[str, list]:
    """
    Get DFT and predicted formation energies for nitrides (PBE(+U)).

    Parameters
    ----------
    build_results
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, list]
        Dictionary of DFT and predicted formation energies.
    """
    result_formation_energies, _, _ = build_results_pbe()
    return result_formation_energies


@pytest.fixture
def formation_energy_pbesol_mae(formation_energies_pbesol) -> dict[str, float]:
    """
    Get mean absolute error for split-vancacy formation energies compared to PBEsol.

    Parameters
    ----------
    formation_energies_pbesol
        Dictionary of DFT and predicted formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of formation energy MAEs for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            formation_energies_pbesol["ref"], formation_energies_pbesol[model_name]
        )
    return results


@pytest.fixture
def formation_energy_pbe_mae(formation_energies_pbe) -> dict[str, float]:
    """
    Get mean absolute error for split-vancacy formation energies compared to PBE(+U).

    Parameters
    ----------
    formation_energies_pbesol
        Dictionary of DFT and predicted formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of formation energy MAEs for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            formation_energies_pbe["ref"], formation_energies_pbe[model_name]
        )
    return results


@pytest.fixture
def spearmans_coefficient_pbesol_mean(build_results_pbe) -> dict[str, float]:
    """
    Energy ranking score of PBEsol relaxed structures (oxides).

    Parameters
    ----------
    build_results
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Dictionary of mean Spearman's coefficients for all models.
    """
    _, result_spearmans_coefficient, _ = build_results_pbe

    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.mean(result_spearmans_coefficient[model_name]))
    return results


@pytest.fixture
def spearmans_coefficient_pbe_mean(build_results_pbe) -> dict[str, float]:
    """
    Energy ranking score of PBE relaxed structures (nitrides).

    Parameters
    ----------
    build_results
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Dictionary of mean Spearman's coefficients for all models.
    """
    _, result_spearmans_coefficient, _ = build_results_pbe

    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.mean(result_spearmans_coefficient[model_name]))
    return results


@pytest.fixture
def rmsd_pbesol_mean(build_results_pbesol) -> dict[str, float]:
    """
    Get RMSD between PBEsol and MLIP relaxed structures (oxides).

    Parameters
    ----------
    build_results
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Dictionary of mean relaxed structure RMSDs for all models.
    """
    _, _, result_rmsd = build_results_pbesol

    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.mean(result_rmsd[model_name]))
    return results


@pytest.fixture
def rmsd_pbesol_mean(build_results_pbesol) -> dict[str, float]:
    """
    Get RMSD between PBE and MLIP relaxed structures (nitrides).

    Parameters
    ----------
    build_results
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Dictionary of mean relaxed structure RMSDs for all models.
    """
    _, _, result_rmsd = build_results_pbesol()

    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.mean(result_rmsd[model_name]))
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "split_vacancy_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    formation_energy_pbesol_mae: dict[str, float],
    spearmans_coefficient_pbesol_mean: dict[str, float],
    rmsd_pbesol_mean: dict[str, float],
    formation_energy_pbe_mae: dict[str, float],
    spearmans_coefficient_pbe_mean: dict[str, float],
    rmsd_pbe_mean: dict[str, float],
) -> dict[str, dict]:
    """
    Get all new benchmark metrics.

    Parameters
    ----------
    formation_energy_dft_mae
        Split vancancy formation energy MAE for all models.
    spearmans_coefficient_dft_mean
        Spearman's coefficient mean for all models.
    rmsd_dft_mean
        RMSD for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    # print(formation_energy_dft_mae)
    return {
        "MAE (PBEsol)": formation_energy_pbesol_mae,
        "Spearman's (PBEsol)": spearmans_coefficient_pbesol_mean,
        "RMSD (PBEsol)": rmsd_pbesol_mean,
        "MAE (PBE)": formation_energy_pbe_mae,
        "Spearman's (PBE)": spearmans_coefficient_pbe_mean,
        "RMSD (PBE)": rmsd_pbe_mean,
    }


def test_new_benchmark(metrics: dict[str, dict]) -> None:
    """
    Run new benchmark analysis.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
