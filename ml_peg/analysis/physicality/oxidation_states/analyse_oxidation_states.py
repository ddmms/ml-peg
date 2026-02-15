"""Analyse aqueous Iron Chloride oxidation states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

# MODELS = get_model_names(current_models)

MODELS = ["mace-mp-0b3", "omol"]

CALC_PATH = CALCS_ROOT / "physicality" / "oxidation_states" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "oxidation_states"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

IRON_SALTS = ["Fe2Cl", "Fe3Cl"]
TESTS = ["Fe-O RDF Peak Split", "Peak Within Experimental Ref"]
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

    for salt in IRON_SALTS:
        rdf_file = CALC_PATH / f"O-Fe_{salt}_{model}.rdf"

        fe_o_rdf = np.loadtxt(rdf_file)
        r = list(fe_o_rdf[:, 0])
        g_r = list(fe_o_rdf[:, 1])

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
        title="Fe-O RDF",
        x_label="R / &Aring;",
        y_label="Fe-O G(r)",
        show_line=True,
        show_markers=False,
        highlight_area=True,
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
        peak_position = {}
        results = get_rdf_results(model)
        plot_rdfs(model, results)

        for salt in IRON_SALTS:
            r = results[salt][0]
            g_r = results[salt][1]
            peak_position[salt] = r[g_r.index(max(g_r))]

        peak_difference = abs(peak_position["Fe2Cl"] - peak_position["Fe3Cl"])

        oxidation_state_passfail["Fe-O RDF Peak Split"][model] = 0.0
        oxidation_state_passfail["Peak Within Experimental Ref"][model] = 0.0

        if peak_difference > 0.1:
            oxidation_state_passfail["Fe-O RDF Peak Split"][model] = 1.0

            if fe_2_ref[0] <= peak_position["Fe2Cl"] <= fe_2_ref[1]:
                oxidation_state_passfail["Peak Within Experimental Ref"][model] += 0.5

            if fe_3_ref[0] <= peak_position["Fe3Cl"] <= fe_3_ref[1]:
                oxidation_state_passfail["Peak Within Experimental Ref"][model] += 0.5

    return oxidation_state_passfail


@pytest.fixture
@build_table(
    filename=OUT_PATH / "oxidation_states_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
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
