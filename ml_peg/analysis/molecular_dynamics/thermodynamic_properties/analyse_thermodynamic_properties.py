"""Analyse the organic liquid thermodynamic properties benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.molecular_dynamics.thermodynamic_properties import (
    analyse_liquid,
)
from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
    maze,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "thermodynamic_properties" / "outputs"

OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "thermodynamic_properties"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*-liq.traj",
    index=0,
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
    include_filenames=True,
)


PROPERTIES = {
    "density": {
        "reference": "exp_density",
        "title": "Density",
        "unit": "g/cm^3",
    },
    "cp": {
        "reference": "exp_cp",
        "title": "Heat capacity at constant pressure",
        "unit": "J/(mol K)",
    },
    "compressibility": {
        "reference": "exp_compressibility",
        "title": "Isothermal compressibility",
        "unit": "1/GPa",
    },
    "alpha": {
        "reference": "exp_alpha",
        "title": "Thermal expansion coefficient",
        "unit": "1e-3/K",
    },
    "evaporation_enthalpy": {
        "reference": "exp_evaporation_enthalpy",
        "title": "Evaporation enthalpy",
        "unit": "kJ/mol",
    },
}


@pytest.fixture
def thermodynamic_properties(
    block_size: int,
    equilib_time_ps: float,
) -> dict[str, dict]:
    """
    Analyse thermodynamic properties for all systems and models.

    Parameters
    ----------
    block_size
        The size of blocks used for error estimate.
    equilib_time_ps
        The initial time (in ps) that is skipped in
        the analysis.

    Returns
    -------
    dict[str, dict]
        Reference values, predictions, and statistical uncertainties.
    """
    results = {
        property_name: {
            "ref": [],
            **{
                model_name: {
                    "value": [],
                    "stderr": [],
                }
                for model_name in MODELS
            },
        }
        for property_name in PROPERTIES
    }

    ref_stored = False

    for model_name in MODELS:
        for label in INFO["filenames"]:
            label_gas = label.removesuffix("-liq") + "-gas"

            xyz_file = CALC_PATH / model_name / f"{label}.xyz"
            xyz_gas_file = CALC_PATH / model_name / f"{label_gas}.xyz"
            log_file = CALC_PATH / model_name / f"{label}.log"
            log_file_gas = CALC_PATH / model_name / f"{label_gas}.log"

            atoms = read(xyz_file)
            atoms_gas = read(xyz_gas_file)

            calculated = analyse_liquid(
                log_file_liq=log_file,
                log_file_gas=log_file_gas,
                temperature=atoms.info["exp_temperature"],
                pressure=atoms.info["exp_pressure"],
                n_molecules=atoms.info["n_molecules"],
                block_size=block_size,
                equilib_time_ps=equilib_time_ps,
            )

            for property_name, (value, stderr) in calculated.items():
                results[property_name][model_name]["value"].append(value)
                results[property_name][model_name]["stderr"].append(stderr)

                if not ref_stored:
                    reference_key = PROPERTIES[property_name]["reference"]
                    results[property_name]["ref"].append(atoms.info[reference_key])

            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)
            write(structs_dir / f"{label_gas}.xyz", atoms_gas)

        ref_stored = True

    return results


def get_property_results(
    results: dict[str, dict],
    property_name: str,
) -> dict[str, list]:
    """
    Extract parity-plot data for one property.

    Parameters
    ----------
    results
        Thermodynamic analysis results.
    property_name
        Name of the property to extract.

    Returns
    -------
    dict[str, list]
        Reference and predicted values for each model.
    """
    return {
        "ref": results[property_name]["ref"],
        **{
            model_name: results[property_name][model_name]["value"]
            for model_name in MODELS
        },
    }


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_density.json",
    title="Density",
    x_label="Predicted density / (g / L)",
    y_label="Reference density / (g / L)",
    hoverdata={"Labels": INFO["filenames"]},
    use_plotly_autorange=False,
)
def density_results(thermodynamic_properties) -> dict[str, list]:
    """
    Return density results for plotting.

    Parameters
    ----------
    thermodynamic_properties
        Thermodynamic analysis results.

    Returns
    -------
    dict[str, list]
        Reference and predicted values for each model.
    """
    return get_property_results(
        thermodynamic_properties,
        "density",
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_cp.json",
    title="Heat capacity at constant pressure",
    x_label="Predicted Cp / (J / mol / K)",
    y_label="Reference Cp / (J / mol / K)",
    hoverdata={"Labels": INFO["filenames"]},
    use_plotly_autorange=False,
)
def cp_results(thermodynamic_properties) -> dict[str, list]:
    """
    Return constant-pressure heat capacities for plotting.

    Parameters
    ----------
    thermodynamic_properties
        Thermodynamic analysis results.

    Returns
    -------
    dict[str, list]
        Reference and predicted values for each model.
    """
    return get_property_results(
        thermodynamic_properties,
        "cp",
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_evaporation_enthalpy.json",
    title="Evaporation enthalpy",
    x_label="Predicted evaporation enthalpy / (kJ / mol)",
    y_label="Reference evaporation enthalpy / (kJ / mol)",
    hoverdata={"Labels": INFO["filenames"]},
    use_plotly_autorange=False,
)
def evaporation_enthalpy_results(thermodynamic_properties) -> dict[str, list]:
    """
    Return constant-pressure heat capacities for plotting.

    Parameters
    ----------
    thermodynamic_properties
        Thermodynamic analysis results.

    Returns
    -------
    dict[str, list]
        Reference and predicted values for each model.
    """
    return get_property_results(
        thermodynamic_properties,
        "evaporation_enthalpy",
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_compressibility.json",
    title="Isothermal compressibility",
    x_label="Predicted compressibility / (1/GPa)",
    y_label="Reference compressibility / (1/GPa)",
    hoverdata={"Labels": INFO["filenames"]},
    use_plotly_autorange=False,
)
def compressibility_results(
    thermodynamic_properties,
) -> dict[str, list]:
    """
    Return compressibilities for plotting.

    Parameters
    ----------
    thermodynamic_properties
        Thermodynamic analysis results.

    Returns
    -------
    dict[str, list]
        Reference and predicted values for each model.
    """
    return get_property_results(
        thermodynamic_properties,
        "compressibility",
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_alpha.json",
    title="Thermal expansion coefficient",
    x_label="Predicted alpha / (1e-3 / K)",
    y_label="Reference alpha / (1e-3 / K)",
    hoverdata={"Labels": INFO["filenames"]},
    use_plotly_autorange=False,
)
def alpha_results(thermodynamic_properties) -> dict[str, list]:
    """
    Return thermal expansion coefficients for plotting.

    Parameters
    ----------
    thermodynamic_properties
        Thermodynamic analysis results.

    Returns
    -------
    dict[str, list]
        Reference and predicted values for each model.
    """
    return get_property_results(
        thermodynamic_properties,
        "alpha",
    )


@pytest.fixture
def get_metrics(
    thermodynamic_properties,
) -> dict[str, dict[str, float]]:
    """
    Compute thermodynamic benchmark metrics.

    Parameters
    ----------
    thermodynamic_properties
        Analysed thermodynamic properties.

    Returns
    -------
    dict[str, dict[str, float]]
        MAE and MAZE values for each property and model.
    """
    results = {}

    for property_name in PROPERTIES:
        ref = np.asarray(thermodynamic_properties[property_name]["ref"])

        for model_name in MODELS:
            pred = np.asarray(
                thermodynamic_properties[property_name][model_name]["value"]
            )
            stderr = np.asarray(
                thermodynamic_properties[property_name][model_name]["stderr"]
            )

            results.setdefault(
                f"{property_name}_MAE",
                {},
            )[model_name] = mae(
                ref,
                pred,
            )

            results.setdefault(
                f"{property_name}_MAZE",
                {},
            )[model_name] = maze(
                ref,
                pred,
                stderr,
            )

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "thermodynamic_properties_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    get_metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Return all benchmark metrics.

    Parameters
    ----------
    get_metrics
        Thermodynamic metrics for all models.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric names and values for all models.
    """
    return get_metrics


@pytest.mark.framework("mace-off-24")
def test_thermodynamic_properties(
    metrics: dict[str, dict[str, float]],
    density_results,
    cp_results,
    evaporation_enthalpy_results,
    compressibility_results,
    alpha_results,
) -> None:
    """
    Run the organic liquids thermodynamic properties benchmark.

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
    """
    return
