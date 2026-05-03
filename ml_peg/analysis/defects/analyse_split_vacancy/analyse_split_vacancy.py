"""Analyse split vacancy benchmark."""

from __future__ import annotations

from pathlib import Path
import shutil

from ase.io import read
import numpy as np
import pytest
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from ml_peg.analysis.utils.decorators import build_table, plot_parity, plot_violin
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
STOL = 0.25  # normalised max-distance threshold for structure matching
CALC_PATH = CALCS_ROOT / "defects" / "split_vacancy" / "outputs"
CALC_PATH_PBESOL = CALC_PATH / "pbesol"  # oxides
CALC_PATH_PBE = CALC_PATH / "pbe"  # nitrides
OUT_PATH = APP_ROOT / "data" / "defects" / "split_vacancy"

print(f"Copying xyz files from {CALC_PATH} to {OUT_PATH} for flask app.")
shutil.copytree(CALC_PATH, OUT_PATH, dirs_exist_ok=True)

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_hoverdata(functional_path: Path) -> tuple[list, list, list]:
    """
    Get hover data.

    Parameters
    ----------
    functional_path
        Path to data.

    Returns
    -------
    tuple[list, list, list ]
        Tuple of Materials Project IDs, bulk formulae and vacant cations.
    """
    mp_ids = []
    formulae = []
    vacant_cations = []

    model_dir = functional_path / MODELS[0]
    for material_dir in model_dir.iterdir():
        split_dir_name = material_dir.stem.split("-")
        bulk_formula = split_dir_name[0]
        mp_id = f"mp-{split_dir_name[-1]}"

        cation_dirs = [
            p for p in material_dir.iterdir() if p.is_dir()
        ]  # skip pristine supercell.xyz files if present (not used)

        for cation_dir in cation_dirs:
            vacant_cation = cation_dir.stem

            mp_ids.append(mp_id)
            formulae.append(bulk_formula)
            vacant_cations.append(vacant_cation)

    return {
        "Materials Project ID": mp_ids,
        "Formula": formulae,
        "Vacant Cation": vacant_cations,
    }


FORMATION_HOVERDATA_PBE = get_hoverdata(CALC_PATH_PBE)
FORMATION_HOVERDATA_PBESOL = get_hoverdata(CALC_PATH_PBESOL)


def get_max_dist_hoverdata(functional_path: Path) -> dict[str, list]:
    """
    Get per-structure hover data for max_dist violin plots.

    Reads xyz files to account for multiple initial structures per
    material-cation pair.

    Parameters
    ----------
    functional_path
        Path to data.

    Returns
    -------
    dict[str, list]
        Hover data aligned with per-structure max_dist values.
    """
    mp_ids: list[str] = []
    formulae: list[str] = []
    vacant_cations: list[str] = []
    vacancy_types: list[str] = []
    frames_ids: list[int] = []

    model_dir = functional_path / MODELS[0]
    for material_dir in model_dir.iterdir():
        split_dir_name = material_dir.stem.split("-")
        bulk_formula = split_dir_name[0]
        mp_id = f"mp-{split_dir_name[-1]}"

        cation_dirs = [p for p in material_dir.iterdir() if p.is_dir()]

        for cation_dir in cation_dirs:
            cation = cation_dir.stem
            nv_xyz_path = cation_dir / "normal_vacancy.xyz"
            sv_xyz_path = cation_dir / "split_vacancy.xyz"

            if not (nv_xyz_path.exists() and sv_xyz_path.exists()):
                continue

            nv_n = len(read(nv_xyz_path, ":"))
            sv_n = len(read(sv_xyz_path, ":"))

            mp_ids.extend([mp_id] * (nv_n + sv_n))
            formulae.extend([bulk_formula] * (nv_n + sv_n))
            vacant_cations.extend([cation] * (nv_n + sv_n))
            vacancy_types.extend(["NV"] * nv_n + ["SV"] * sv_n)
            frames_ids.extend(list(range(nv_n)) + list(range(sv_n)))

    return {
        "Materials Project ID": mp_ids,
        "Formula": formulae,
        "Vacant Cation": vacant_cations,
        "Type": vacancy_types,
        "Frame ID": frames_ids,
    }


MAX_DIST_HOVERDATA_PBE = get_max_dist_hoverdata(CALC_PATH_PBE)
MAX_DIST_HOVERDATA_PBESOL = get_max_dist_hoverdata(CALC_PATH_PBESOL)


def build_results(
    functional_path,
) -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
    """
    Iterate through bulk-cation pairs calculating results.

    Parameters
    ----------
    functional_path
        Path to data.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, float]]
        Tuple of metrics.
    """
    print(f"Analysing {functional_path.stem} calculations.")

    result_formation_energy = {"ref": []} | {
        mlip: [] for mlip in MODELS
    }  # formation energy for every material-cation pair
    result_spearmans_coefficient = {
        mlip: [] for mlip in MODELS
    }  # spearmans coefficient for every material-cation pair
    result_rmsd = {
        mlip: [] for mlip in MODELS
    }  # normalized RMSD error for every material-cation pair
    result_max_dist = {
        mlip: [] for mlip in MODELS
    }  # normalized max_dist for every material-cation pair
    result_match = {mlip: [] for mlip in MODELS}  # if structures relaxing to same state

    ref_stored = False

    for model_name in tqdm(MODELS):
        model_dir = functional_path / model_name

        if not model_dir.exists():
            continue

        for material_dir in tqdm(list(model_dir.iterdir()), leave=False):
            cation_dirs = [
                p for p in material_dir.iterdir() if p.is_dir()
            ]  # skip pristine supercell.xyz files if present (not used)

            for cation_dir in cation_dirs:
                nv_xyz_path = cation_dir / "normal_vacancy.xyz"
                sv_xyz_path = cation_dir / "split_vacancy.xyz"

                if not (nv_xyz_path.exists() and sv_xyz_path.exists()):
                    raise ValueError(
                        f"Missing xyz file(s) in {cation_dir}. Expected both "
                        f"normal_vacancy.xyz and split_vacancy.xyz to exist."
                    )

                nv_atoms_list = read(nv_xyz_path, ":")
                sv_atoms_list = read(sv_xyz_path, ":")

                # Load reference data

                ref_nv_energies = [float(at.info["ref_energy"]) for at in nv_atoms_list]
                ref_sv_energies = [float(at.info["ref_energy"]) for at in sv_atoms_list]

                if not ref_stored:
                    ref_sv_formation_energy = min(ref_sv_energies) - min(
                        ref_nv_energies
                    )
                    result_formation_energy["ref"].append(ref_sv_formation_energy)

                match_list = []
                rmsd_list = []
                max_dist_list = []

                nv_initial_energies = []
                nv_relaxed_energies = []
                for nv_atoms in nv_atoms_list:
                    match = nv_atoms.info["ref_max_distance"] < STOL
                    match_list.append(match)
                    if match:
                        nv_relaxed_energies.append(nv_atoms.info["relaxed_energy"])

                    rmsd_list.append(nv_atoms.info["ref_rmsd"])
                    max_dist_list.append(nv_atoms.info["ref_max_distance"])
                    nv_initial_energies.append(nv_atoms.info["initial_energy"])

                sv_initial_energies = []
                sv_relaxed_energies = []
                for sv_atoms in sv_atoms_list:
                    match = sv_atoms.info["ref_max_distance"] < STOL
                    match_list.append(match)
                    if match:
                        sv_relaxed_energies.append(sv_atoms.info["relaxed_energy"])

                    rmsd_list.append(sv_atoms.info["ref_rmsd"])
                    max_dist_list.append(sv_atoms.info["ref_max_distance"])
                    sv_initial_energies.append(sv_atoms.info["initial_energy"])

                sv_formation_energy = min(sv_relaxed_energies, default=np.nan) - min(
                    nv_relaxed_energies, default=np.nan
                )
                spearmans_coefficient = spearmanr(
                    nv_initial_energies + sv_initial_energies,
                    ref_nv_energies + ref_sv_energies,
                ).statistic

                # add metrics to dicts
                result_formation_energy[model_name].append(sv_formation_energy)
                result_spearmans_coefficient[model_name].append(spearmans_coefficient)
                result_rmsd[model_name].extend(rmsd_list)
                result_max_dist[model_name].extend(max_dist_list)
                result_match[model_name].extend(match_list)

        ref_stored = True

    return (
        result_formation_energy,
        result_spearmans_coefficient,
        result_rmsd,
        result_match,
        result_max_dist,
    )


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
    hoverdata=FORMATION_HOVERDATA_PBESOL,
)
def formation_energies_pbesol(build_results_pbesol) -> dict[str, list]:
    """
    Get DFT and predicted formation energies for oxides (PBEsol).

    Parameters
    ----------
    build_results_pbesol
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, list]
        Dictionary of DFT and predicted formation energies.
    """
    result_formation_energies, _, _, _, _ = build_results_pbesol
    return result_formation_energies


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_formation_energies_pbe.json",
    title="Split vacancy (Nitrides, PBE(+U))",
    x_label="Predicted Split Vacancy Formation Energy / eV",
    y_label="DFT Split Vacancy Formation Energy / eV",
    hoverdata=FORMATION_HOVERDATA_PBE,
)
def formation_energies_pbe(build_results_pbe) -> dict[str, list]:
    """
    Get DFT and predicted formation energies for nitrides (PBE(+U)).

    Parameters
    ----------
    build_results_pbe
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, list]
        Dictionary of DFT and predicted formation energies.
    """
    result_formation_energies, _, _, _, _ = build_results_pbe
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
        formation_energies_ref = formation_energies_pbesol["ref"].copy()
        formation_energies_model = formation_energies_pbesol[model_name].copy()

        nan_mask = ~np.isnan(formation_energies_model)
        formation_energies_ref = np.array(formation_energies_ref)[nan_mask]
        formation_energies_model = np.array(formation_energies_model)[nan_mask]

        results[model_name] = mae(formation_energies_ref, formation_energies_model)
    return results


@pytest.fixture
def formation_energy_pbe_mae(formation_energies_pbe) -> dict[str, float]:
    """
    Get mean absolute error for split-vancacy formation energies compared to PBE(+U).

    Parameters
    ----------
    formation_energies_pbe
        Dictionary of DFT and predicted formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of formation energy MAEs for all models.
    """
    results = {}
    for model_name in MODELS:
        formation_energies_ref = formation_energies_pbe["ref"].copy()
        formation_energies_model = formation_energies_pbe[model_name].copy()

        nan_mask = ~np.isnan(formation_energies_model)
        formation_energies_ref = np.array(formation_energies_ref)[nan_mask]
        formation_energies_model = np.array(formation_energies_model)[nan_mask]

        results[model_name] = mae(formation_energies_ref, formation_energies_model)
    return results


@pytest.fixture
def spearmans_coefficient_pbesol_mean(build_results_pbesol) -> dict[str, float]:
    """
    Energy ranking score of PBEsol relaxed structures (oxides).

    Parameters
    ----------
    build_results_pbesol
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Dictionary of mean Spearman's coefficients for all models.
    """
    _, result_spearmans_coefficient, _, _, _ = build_results_pbesol

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
    build_results_pbe
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Dictionary of mean Spearman's coefficients for all models.
    """
    _, result_spearmans_coefficient, _, _, _ = build_results_pbe

    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.mean(result_spearmans_coefficient[model_name]))
    return results


# @pytest.fixture
# @plot_violin(
#     title="RMSD Distribution (Oxides, PBEsol)",
#     y_label="RMSD / Å",
#     filename=OUT_PATH / "figure_rmsd_pbesol.json",
# )
# def rmsd_pbesol_dist(build_results_pbesol) -> dict[str, list]:
#     _, _, result_rmsd, _, _ = build_results_pbesol
#     return result_rmsd


@pytest.fixture
def rmsd_pbesol_mean(build_results_pbesol) -> dict[str, float]:
    """
    Get RMSD between PBEsol and MLIP relaxed structures (oxides).

    Parameters
    ----------
    build_results_pbesol
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Mean RMSD between MLIP and DFT relaxed structure that match.
    """
    _, _, result_rmsd, _, _ = build_results_pbesol
    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.nanmean(result_rmsd[model_name]))
    return results


# @pytest.fixture
# @plot_violin(
#     title="RMSD Distribution (Nitrides, PBE(+U))",
#     y_label="RMSD / Å",
#     filename=OUT_PATH / "figure_rmsd_pbe.json",
# )
# def rmsd_pbe_dist(build_results_pbe) -> dict[str, list]:
#     _, _, result_rmsd, _, _ = build_results_pbe
#     return result_rmsd


@pytest.fixture
def rmsd_pbe_mean(build_results_pbe) -> dict[str, float]:
    """
    Get RMSD between PBE and MLIP relaxed structures (nitrides).

    Parameters
    ----------
    build_results_pbe
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Mean RMSD between MLIP and DFT relaxed structures that match.
    """
    _, _, result_rmsd, _, _ = build_results_pbe
    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.nanmean(result_rmsd[model_name]))
    return results


@pytest.fixture
def match_pbesol_rate(build_results_pbesol) -> dict[str, float]:
    """
    Get RMSD between PBEsol and MLIP relaxed structures (oxides).

    Parameters
    ----------
    build_results_pbesol
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Rate of MLIP relaxing to same structure as DFT.
    """
    _, _, _, result_match, _ = build_results_pbesol

    results = {}
    for model_name in MODELS:
        results[model_name] = np.mean(result_match[model_name])
    return results


@pytest.fixture
def match_pbe_rate(build_results_pbe) -> dict[str, float]:
    """
    Get RMSD between PBE and MLIP relaxed structures (oxides).

    Parameters
    ----------
    build_results_pbe
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, float]
        Rate of MLIP relaxing to same structure as DFT.
    """
    _, _, _, result_match, _ = build_results_pbe

    results = {}
    for model_name in MODELS:
        results[model_name] = np.mean(result_match[model_name])
    return results


@pytest.fixture
@plot_violin(
    title="Max Distance Distribution (Oxides, PBEsol)",
    y_label="Max Distance / Å",
    hoverdata=MAX_DIST_HOVERDATA_PBESOL,
    filename=OUT_PATH / "figure_max_dist_pbesol.json",
)
def max_dist_pbesol_dist(build_results_pbesol) -> dict[str, list]:
    """
    Get max dist distributions between PBEsol and MLIP relaxed structures (oxides).

    Parameters
    ----------
    build_results_pbesol
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, list]
        Per-model lists of max_dist values.
    """
    _, _, _, _, result_max_dist = build_results_pbesol
    return result_max_dist


@pytest.fixture
def max_dist_pbesol_mean(max_dist_pbesol_dist) -> dict[str, float]:
    """
    Get mean max dist between PBEsol and MLIP relaxed structures (oxides).

    Parameters
    ----------
    max_dist_pbesol_dist
        Per-model lists of max_dist values.

    Returns
    -------
    dict[str, float]
        Mean max_dist between MLIP and DFT relaxed structure that match.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.nanmean(max_dist_pbesol_dist[model_name]))
    return results


@pytest.fixture
@plot_violin(
    title="Max Distance Distribution (Nitrides, PBE(+U))",
    y_label="Max Distance / Å",
    hoverdata=MAX_DIST_HOVERDATA_PBE,
    filename=OUT_PATH / "figure_max_dist_pbe.json",
)
def max_dist_pbe_dist(build_results_pbe) -> dict[str, list]:
    """
    Get max dist distributions between PBE and MLIP relaxed structures (nitrides).

    Parameters
    ----------
    build_results_pbe
        Tuple of results dictionaries.

    Returns
    -------
    dict[str, list]
        Per-model lists of max_dist values.
    """
    _, _, _, _, result_max_dist = build_results_pbe
    return result_max_dist


@pytest.fixture
def max_dist_pbe_mean(max_dist_pbe_dist) -> dict[str, float]:
    """
    Get mean max dist between PBE and MLIP relaxed structures (nitrides).

    Parameters
    ----------
    max_dist_pbe_dist
        Per-model lists of max_dist values.

    Returns
    -------
    dict[str, float]
        Mean max_dist between MLIP and DFT relaxed structure that match.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = float(np.nanmean(max_dist_pbe_dist[model_name]))
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
    match_pbesol_rate: dict[str, float],
    max_dist_pbesol_mean: dict[str, float],
    formation_energy_pbe_mae: dict[str, float],
    spearmans_coefficient_pbe_mean: dict[str, float],
    match_pbe_rate: dict[str, float],
    max_dist_pbe_mean: dict[str, float],
) -> dict[str, dict]:
    """
    Get all new benchmark metrics.

    Parameters
    ----------
    formation_energy_pbesol_mae
        Split vacancy formation energy MAE (oxides, PBEsol) for all models.
    spearmans_coefficient_pbesol_mean
        Mean Spearman's rank correlation (oxides, PBEsol) for all models.
    match_pbesol_rate
        Rate of MLIP relaxing to same structure as PBEsol.
    max_dist_pbesol_mean
        Mean max distance between MLIP and PBEsol relaxed structures.
    formation_energy_pbe_mae
        Split vacancy formation energy MAE (nitrides, PBE(+U)) for all models.
    spearmans_coefficient_pbe_mean
        Mean Spearman's rank correlation (nitrides, PBE(+U)) for all models.
    match_pbe_rate
        Rate of MLIP relaxing to same structure as PBE.
    max_dist_pbe_mean
        Mean max distance between MLIP and PBE relaxed structures.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE (Oxides)": formation_energy_pbesol_mae,
        "Spearman's (Oxides)": spearmans_coefficient_pbesol_mean,
        # "RMSD (PBEsol)": rmsd_pbesol_mean,
        "Match Rate (Oxides)": match_pbesol_rate,
        "Max Dist (Oxides)": max_dist_pbesol_mean,
        "MAE (Nitrides)": formation_energy_pbe_mae,
        "Spearman's (Nitrides)": spearmans_coefficient_pbe_mean,
        # "RMSD (PBE)": rmsd_pbe_mean,
        "Match Rate (Nitrides)": match_pbe_rate,
        "Max Dist (Nitrides)": max_dist_pbe_mean,
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
