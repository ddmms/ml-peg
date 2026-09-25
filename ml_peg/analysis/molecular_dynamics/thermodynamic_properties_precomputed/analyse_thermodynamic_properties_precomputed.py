"""Analyse the organic liquid thermodynamic precomputed properties benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_peg.analysis.molecular_dynamics.thermodynamic_properties.utils import (
    build_table_factory,
    detailed_results_factory,
    get_metrics_factory,
    get_property_results_factory,
    get_struct_info_thermodynamic_properties,
    property_fixture_factory,
    thermodynamic_properties_factory,
)
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)

# Let's distinguish the output artefacts based on wether we
# are using the precomputed logs or not.

CALC_PATH = (
    CALCS_ROOT
    / "molecular_dynamics"
    / "thermodynamic_properties_precomputed"
    / "outputs"
)

OUT_PATH = (
    APP_ROOT / "data" / "molecular_dynamics" / "thermodynamic_properties_precomputed"
)

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info_thermodynamic_properties(CALC_PATH, OUT_PATH)

thermodynamic_properties = thermodynamic_properties_factory(
    MODELS, INFO, CALC_PATH, OUT_PATH
)

get_property_results = get_property_results_factory(MODELS)


# factory-generated fixtures for the parity plots
density_results = property_fixture_factory("density", INFO, OUT_PATH, MODELS)
cp_results = property_fixture_factory("cp", INFO, OUT_PATH, MODELS)
evaporation_enthalpy_results = property_fixture_factory(
    "evaporation_enthalpy", INFO, OUT_PATH, MODELS
)
compressibility_results = property_fixture_factory(
    "compressibility", INFO, OUT_PATH, MODELS
)
alpha_results = property_fixture_factory("alpha", INFO, OUT_PATH, MODELS)

get_metrics = get_metrics_factory(MODELS)

metrics = build_table_factory(
    OUT_PATH, DEFAULT_TOOLTIPS, DEFAULT_THRESHOLDS, D3_MODEL_NAMES
)

detailed_results_output = detailed_results_factory(MODELS, OUT_PATH)


@pytest.mark.framework("mace-off-24")
def test_thermodynamic_properties_precomputed(
    metrics: dict[str, dict[str, float]],
    density_results,
    cp_results,
    evaporation_enthalpy_results,
    compressibility_results,
    alpha_results,
    detailed_results_output,
) -> None:
    """
    Run the organic liquids thermodynamic precomputed properties benchmark.

    Parameters
    ----------
    metrics
        Benchmark metrics for all models.
    density_results
        Density results used to generate the parity plot.
    cp_results
        Heat capacity results used to generate the parity plot.
    evaporation_enthalpy_results
        Evaporation enthalpy results used to generate the parity plot.
    compressibility_results
        Isothermal compressibility results used to generate the parity plot.
    alpha_results
        Thermal expansion coefficient results used to generate the parity plot.
    detailed_results_output
        Detailed results for each chemical.
    """
    return
