"""Analyse aqueous Iron Chloride oxidation states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "physicality" / "oxidation_states" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "oxidation_states"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

IRON_SALTS = ["Fe2Cl", "Fe3Cl"]
TESTS = [
    "Fe-O RDF Peak Split",
    "Fe +2 Peak Experimental Ref Deviation",
    "Fe +3 Peak Experimental Ref Deviation",
]
REF_PEAK_RANGE = {
    "Fe<sup>+2</sup><br>Ref": [2.0, 2.2],
    "Fe<sup>+3</sup><br>Ref": [1.9, 2.0],
}


def get_rdf_results(
    model: str,
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Get a model's Fe-O RDFs for the aqueous Fe2Cl and Fe3Cl MD.

    Parameters
    ----------
    model
        Name of MLIP.

    Returns
    -------
    results
        RDF Radii and intensities for the aqueous Fe2Cl and Fe3Cl systems.
    """
    results = {salt: [] for salt in IRON_SALTS}

    model_calc_path = CALC_PATH / model

    for salt in IRON_SALTS:
        rdf_file = model_calc_path / f"O-Fe_{salt}_{model}.rdf"

        fe_o_rdf = np.loadtxt(rdf_file)
        r = fe_o_rdf[:, 0]
        g_r = fe_o_rdf[:, 1]

        results[salt].append(r)
        results[salt].append(g_r)

    return results


def plot_rdfs(model: str, results: dict[str, tuple[list[float], list[float]]]) -> None:
    """
    Plot Fe-O RDFs.

    Parameters
    ----------
    model
        Name of MLIP.
    results
        RDF Radii and intensities for the aqueous Fe2Cl and Fe3Cl systems.
    """

    @plot_scatter(
        filename=OUT_PATH / f"Fe-O_{model}_RDF_scatter.json",
        title=f"<b>{model} MD</b>",
        x_label="r [Å]",
        y_label="Fe-O G(r)",
        show_line=True,
        show_markers=False,
        highlight_range=REF_PEAK_RANGE,
    )
    def plot_result() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot the RDFs.

        Returns
        -------
        model_results
            Dictionary of model Fe-O RDFs for the aqueous Fe2Cl and Fe3Cl systems.
        """
        return results

    plot_result()


@pytest.fixture
def get_oxidation_states_passfail() -> dict[str, dict]:
    """
    Test whether model RDF peaks are split and they fall within the reference range.

    Returns
    -------
    oxidation_states_passfail
        Dictionary of pass fail per model.
    """
    oxidation_state_passfail = {test: {} for test in TESTS}

    fe_2_ref = [2.0, 2.2]
    fe_3_ref = [1.9, 2.0]

    for model in MODELS:
        results = get_rdf_results(model)
        plot_rdfs(model, results)

        fe2_r = results[IRON_SALTS[0]][0]
        fe2_g_r = results[IRON_SALTS[0]][1]
        norm_fe2_g_r = fe2_g_r / np.sum(fe2_g_r)
        fe2_peak_pos = fe2_r[np.argmax(fe2_g_r)]

        fe3_r = results[IRON_SALTS[1]][0]
        fe3_g_r = results[IRON_SALTS[1]][1]
        norm_fe3_g_r = fe3_g_r / np.sum(fe3_g_r)
        fe3_peak_pos = fe3_r[np.argmax(fe3_g_r)]

        diff = norm_fe2_g_r - norm_fe3_g_r
        mae = np.sum(np.absolute(diff)) / (np.sum(norm_fe2_g_r) + np.sum(norm_fe3_g_r))

        if mae > 0.2:
            oxidation_state_passfail["Fe-O RDF Peak Split"][model] = 1.0

        else:
            oxidation_state_passfail["Fe-O RDF Peak Split"][model] = 0.0

        oxidation_state_passfail["Fe +2 Peak Experimental Ref Deviation"][model] = (
            normalised_peak_error(fe2_peak_pos, fe_2_ref)
        )
        oxidation_state_passfail["Fe +3 Peak Experimental Ref Deviation"][model] = (
            normalised_peak_error(fe3_peak_pos, fe_3_ref)
        )

    return oxidation_state_passfail


@pytest.fixture
@build_table(
    filename=OUT_PATH / "oxidation_states_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def oxidation_states_passfail_metrics(
    get_oxidation_states_passfail: dict[str, dict],
) -> dict[str, dict]:
    """
    Get all oxidation states pass fail metrics.

    Parameters
    ----------
    get_oxidation_states_passfail
        Dictionary of pass fail per model.

    Returns
    -------
    dict[str, dict]
        Dictionary of pass fail per model.
    """
    return get_oxidation_states_passfail


def test_oxidation_states_passfail_metrics(
    oxidation_states_passfail_metrics: dict[str, dict],
) -> None:
    """
    Run oxidation states test.

    Parameters
    ----------
    oxidation_states_passfail_metrics
        All oxidation states pass fail.
    """
    return


def normalised_peak_error(peak_pos: float, ref_range: list[float, float]) -> float:
    """
    Evaluate normalised peak error compared to reference range.

    Parameters
    ----------
    peak_pos
        Position of the first RDF peak.
    ref_range
        Peak position range from reference.

    Returns
    -------
    float
        Normalised peak deviation from reference range.
    """
    low = ref_range[0]
    high = ref_range[1]
    range_size = high - low
    if peak_pos < low:
        return (peak_pos - low) / range_size
    if peak_pos > high:
        return (peak_pos - high) / range_size
    return 0.0
