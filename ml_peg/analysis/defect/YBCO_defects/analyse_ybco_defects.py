"""
Analyse the YBCO point-defect formation-energy benchmark.

Formation energies (host at fixed volume; mu_O from O2, mu_M from bulk metal):

* vacancy       E_f = E_def - E_perf + mu_removed
* antisite A_B  E_f = E_def - E_perf + mu_removed(B) - mu_added(A)
* interstitial  E_f = E_def - E_perf - mu_added
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, rmse
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "defect" / "YBCO_defects" / "outputs"
OUT_PATH = APP_ROOT / "data" / "defect" / "YBCO_defects"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# CP2K PBE reference formation energies (eV). Reference:
REFERENCE_FE = {
    "O1": 1.37,
    "O2": 1.99,
    "O3": 1.94,
    "O4": 1.64,
    "Cu1": 2.49,
    "Cu2": 1.84,
    "Ba": 6.87,
    "Y": 10.85,
    "Ba_Cu1": -1.16,
    "Ba_Cu2": -0.28,
    "Ba_Y": 4.11,
    "Y_Ba": -2.37,
    "Y_Cu1": -4.81,
    "Y_Cu2": -5.95,
    "Cu_Y": 9.25,
    "Cu_Ba": 5.50,
    "Oint1": 0.83,
    "Oint2": 0.03,
    "Oint3": 1.18,
    "Oint4": 0.03,
    "Oint5": 0.27,
    "Oint6": 1.25,
    "Oint7": 1.46,
}
KIND_TO_CLASS = {"vac": "vacancy", "anti": "antisite", "int": "interstitial"}

# Antisite exchange reactions (Kroger-Vink pairs). The chemical potentials cancel in the
# sum, so these energies are mu-independent: E_react = E(A_B) + E(B_A) - 2 E_perf.
# DFT reference values (eV) from arXiv:2511.22592.
REACTIONS = {
    "Ba_Y + Y_Ba": (("Ba_Y", "Y_Ba"), 1.74),
    "Ba_Cu1 + Cu_Ba": (("Ba_Cu1", "Cu_Ba"), 4.34),
    "Ba_Cu2 + Cu_Ba": (("Ba_Cu2", "Cu_Ba"), 5.21),
    "Y_Cu1 + Cu_Y": (("Y_Cu1", "Cu_Y"), 4.44),
    "Y_Cu2 + Cu_Y": (("Y_Cu2", "Cu_Y"), 3.30),
}
REACTION_NAMES = list(REACTIONS)
REACTION_REF = [REACTIONS[r][1] for r in REACTION_NAMES]


def _name_class(stem: str) -> tuple[str, str]:
    """
    Split a trajectory stem ("vac_O1-traj", "anti_Ba_Cu1-traj") into (name, class).

    Parameters
    ----------
    stem
        Trajectory file stem written by the calc step.

    Returns
    -------
    tuple[str, str]
        Defect name (e.g. "O1", "Ba_Cu1") and class (vacancy/antisite/interstitial).
    """
    core = stem[:-5] if stem.endswith("-traj") else stem
    kind, name = core.split("_", 1)
    return name, KIND_TO_CLASS[kind]


# Defect labels and elements (for filtering), from the mock outputs. Perfect cell and
# references live in _energetics/ and are excluded by the glob.
try:
    INFO = get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="*-traj.extxyz",
        index="-1",
        write_info=True,
        write_structs=False,
        out_path=OUT_PATH,
        include_filenames=True,
    )
except ValueError:
    INFO = {"filenames": [], "elements": []}

DEFECTS, CLASSES = [], []
for _stem in INFO["filenames"]:
    _n, _c = _name_class(_stem)
    DEFECTS.append(_n)
    CLASSES.append(_c)
CLASS = dict(zip(DEFECTS, CLASSES, strict=True))
REF_FLAT = [REFERENCE_FE[d] for d in DEFECTS]


def _element(site: str) -> str:
    """
    Map a site (e.g. 'Cu1', 'O2', 'Ba', 'Y') to its chemical element.

    Parameters
    ----------
    site
        Site possibly carrying a numeric suffix.

    Returns
    -------
    str
        The element symbol.
    """
    return "".join(c for c in site if c.isalpha())


def _energy(traj: Path) -> float:
    """
    Read the final-frame potential energy from a trajectory file.

    Parameters
    ----------
    traj
        Path to the trajectory file.

    Returns
    -------
    float
        Potential energy in eV, or NaN if unavailable.
    """
    if not traj.is_file():
        return np.nan
    try:
        atoms = read(traj, index="-1")
        return float(atoms.get_potential_energy())
    except Exception:  # noqa: BLE001
        return np.nan


def _formation_energies(model_name: str) -> dict[str, float]:
    """
    Compute all defect formation energies for one model.

    Parameters
    ----------
    model_name
        Name of the model whose outputs to read.

    Returns
    -------
    dict[str, float]
        Formation energy (eV) per defect, NaN where inputs are missing.
    """
    model_dir = CALC_PATH / model_name
    energetics = model_dir / "_energetics"
    e_perf = _energy(energetics / "perfect-traj.extxyz")

    # Chemical potentials: mu_O from O2 dimer, mu_M from bulk metal cells.
    mu: dict[str, float] = {}
    for ref, elem in (("O2", "O"), ("Ba", "Ba"), ("Cu", "Cu"), ("Y", "Y")):
        traj = energetics / f"ref_{ref}-traj.extxyz"
        if not traj.is_file():
            mu[elem] = np.nan
            continue
        atoms = read(traj, index="-1")
        try:
            mu[elem] = float(atoms.get_potential_energy()) / len(atoms)
        except Exception:  # noqa: BLE001
            mu[elem] = np.nan

    out: dict[str, float] = {}
    for stem, name, cls in zip(INFO["filenames"], DEFECTS, CLASSES, strict=True):
        e_def = _energy(model_dir / f"{stem}.extxyz")
        delta = e_def - e_perf
        if cls == "vacancy":
            out[name] = delta + mu[_element(name)]
        elif cls == "interstitial":
            out[name] = delta - mu["O"]
        else:  # antisite "A_B": add A, remove B
            added, removed = name.split("_")
            out[name] = delta + mu[_element(removed)] - mu[_element(added)]
    return out


def _reaction_energies(model_name: str) -> dict[str, float]:
    """
    Compute the mu-independent antisite exchange-reaction energies for one model.

    Parameters
    ----------
    model_name
        Name of the model whose outputs to read.

    Returns
    -------
    dict[str, float]
        Reaction energy (eV) per reaction, NaN where inputs are missing.
    """
    model_dir = CALC_PATH / model_name
    e_perf = _energy(model_dir / "_energetics" / "perfect-traj.extxyz")
    out: dict[str, float] = {}
    for name, (pair, _ref) in REACTIONS.items():
        e_a = _energy(model_dir / f"anti_{pair[0]}-traj.extxyz")
        e_b = _energy(model_dir / f"anti_{pair[1]}-traj.extxyz")
        out[name] = e_a + e_b - 2.0 * e_perf
    return out


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_ybco_defects.json",
    title="YBCO defect formation energies",
    x_label="Predicted formation energy / eV",
    y_label="CP2K PBE formation energy / eV",
    hoverdata={"Defect": DEFECTS, "Class": CLASSES},
)
def ybco_defects() -> dict[str, list]:
    """
    Get DFT and predicted defect formation energies, writing per-point structures.

    Returns
    -------
    dict[str, list]
        Reference and predicted formation energies, ordered as ``INFO["filenames"]``.
    """
    results = {"ref": REF_FLAT} | {mlip: [] for mlip in MODELS}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        fe = _formation_energies(model_name)
        for stem, name in zip(INFO["filenames"], DEFECTS, strict=True):
            results[model_name].append(fe.get(name, np.nan))
            # copy structure to app dir for visualisation
            src = model_dir / f"{stem}.extxyz"
            if src.is_file():
                structs_dir = OUT_PATH / model_name
                structs_dir.mkdir(parents=True, exist_ok=True)
                write(structs_dir / f"{name}.xyz", read(src, index="-1"))
    return results


def _class_errors(ybco_defects: dict[str, list], cls: str | None) -> dict[str, float]:
    """
    RMSD of formation energies vs DFT, optionally restricted to one defect class.

    Parameters
    ----------
    ybco_defects
        Reference and predicted formation energies.
    cls
        Defect class to restrict to ("vacancy"/"antisite"/"interstitial"), or None.

    Returns
    -------
    dict[str, float]
        RMSD per model, in eV.
    """
    mask = [cls is None or c == cls for c in CLASSES]
    ref = [v for v, m in zip(ybco_defects["ref"], mask, strict=True) if m]
    results = {}
    for model_name in MODELS:
        pred = [v for v, m in zip(ybco_defects[model_name], mask, strict=True) if m]
        results[model_name] = rmse(ref, pred)
    return results


@pytest.fixture
def ybco_vacancy_errors(ybco_defects) -> dict[str, float]:
    """
    RMSD for vacancies. See :func:`_class_errors`.

    Parameters
    ----------
    ybco_defects
        Reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        RMSD per model, in eV.
    """
    return _class_errors(ybco_defects, "vacancy")


@pytest.fixture
def ybco_antisite_errors(ybco_defects) -> dict[str, float]:
    """
    RMSD for antisites. See :func:`_class_errors`.

    Parameters
    ----------
    ybco_defects
        Reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        RMSD per model, in eV.
    """
    return _class_errors(ybco_defects, "antisite")


@pytest.fixture
def ybco_interstitial_errors(ybco_defects) -> dict[str, float]:
    """
    RMSD for interstitials. See :func:`_class_errors`.

    Parameters
    ----------
    ybco_defects
        Reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        RMSD per model, in eV.
    """
    return _class_errors(ybco_defects, "interstitial")


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_ybco_reactions.json",
    title="YBCO antisite exchange-reaction energies",
    x_label="Predicted reaction energy / eV",
    y_label="CP2K PBE reaction energy / eV",
    hoverdata={"Reaction": REACTION_NAMES},
)
def ybco_reactions() -> dict[str, list]:
    """
    Get DFT and predicted mu-independent antisite reaction energies for all models.

    Returns
    -------
    dict[str, list]
        Reference and predicted reaction energies, ordered as ``REACTION_NAMES``.
    """
    results = {"ref": REACTION_REF} | {mlip: [] for mlip in MODELS}
    for model_name in MODELS:
        re = _reaction_energies(model_name)
        results[model_name] = [re.get(r, np.nan) for r in REACTION_NAMES]
    return results


@pytest.fixture
def ybco_reaction_errors(ybco_reactions) -> dict[str, float]:
    """
    RMSD of the antisite reaction energies vs DFT.

    Parameters
    ----------
    ybco_reactions
        Reference and predicted reaction energies.

    Returns
    -------
    dict[str, float]
        RMSD per model, in eV.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = rmse(ybco_reactions["ref"], ybco_reactions[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ybco_defects_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    ybco_vacancy_errors: dict[str, float],
    ybco_antisite_errors: dict[str, float],
    ybco_interstitial_errors: dict[str, float],
    ybco_reaction_errors: dict[str, float],
) -> dict[str, dict]:
    """
    Get all YBCO defect metrics (per-class formation-energy RMSD + reaction RMSD).

    Parameters
    ----------
    ybco_vacancy_errors
        RMSD for vacancies.
    ybco_antisite_errors
        RMSD for antisites.
    ybco_interstitial_errors
        RMSD for interstitials.
    ybco_reaction_errors
        RMSD for the mu-independent antisite reaction energies.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "RMSD vacancy": ybco_vacancy_errors,
        "RMSD antisite": ybco_antisite_errors,
        "RMSD interstitial": ybco_interstitial_errors,
        "RMSD reactions": ybco_reaction_errors,
    }


def test_ybco_defects(metrics: dict[str, dict]) -> None:
    """
    Run the YBCO defect formation-energy analysis.

    Parameters
    ----------
    metrics
        All YBCO defect formation-energy metrics.
    """
    return
