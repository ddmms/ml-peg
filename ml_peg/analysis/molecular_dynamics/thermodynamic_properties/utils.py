"""Utility functions to process logs and compute thermodynamic properties."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase.io import read, write
import numpy as np
import pytest
import red

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.thermodynamics import (
    density,
    evaporation_enthalpy,
    heat_capacity_cp,
    isothermal_compressibility,
    thermal_expansion,
)
from ml_peg.analysis.utils.utils import (
    get_struct_info,
    mae,
    maze,
)

try:
    from tqdm.auto import tqdm as tqdm
except ModuleNotFoundError:
    tqdm = None

PROPERTIES = {
    "density": {
        "reference": "exp_density",
        "title": "Density",
        "unit": "g/L",
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


def read_property_from_log(
    fname: Path,
    property_name: str,
) -> tuple[np.ndarray, str | None]:
    """
    Read a property time series and unit from a log file.

    Parameters
    ----------
    fname
        Path to the log file.
    property_name
        Name of the property to extract from the log.

    Returns
    -------
    np.ndarray
        Property values recorded.
    str | None
        Unit associated with the property.
    """
    values = []
    unit = None
    last_time_ps = -1.0
    with open(fname) as lines:
        for line in lines:
            items = line.strip().split()
            try:
                time_index = items.index("t:")
                property_index = items.index(f"{property_name}:")

                time_ps = float(items[time_index + 1])
                if time_ps <= last_time_ps:
                    continue
                last_time_ps = time_ps
                value = float(items[property_index + 1])

                if property_index + 2 < len(items):
                    unit = items[property_index + 2]

            except (ValueError, IndexError):
                continue
            values.append(value)
    return np.asarray(values), unit


def detect_equilibration_time(energy, density, time_ps, block_size):
    """
    Detect equilibration time using the RED window method.

    Determines the equilibration point according to the minimum of the squared
    standard error for block-averaged energy and density
    (Clark et al., J. Chem. Theory Comput. 2025,
    doi:10.1021/acs.jctc.4c01359).

    Parameters
    ----------
    energy
        Potential energy time series.
    density
        Density time series.
    time_ps
        Simulation time, in ps, corresponding to each logged sample.
    block_size
        Number of samples in each block used for equilibration detection.

    Returns
    -------
    float
        Estimated equilibration time in ps.
    """
    energy = np.asarray(energy)
    density = np.asarray(density)
    time_ps = np.asarray(time_ps)

    n = min(len(energy), len(density), len(time_ps)) // block_size
    if n < 2:
        raise ValueError("Trajectory is too short for equilibration detection.")

    energy_block = energy[: n * block_size].reshape(n, block_size).mean(axis=1)
    density_block = density[: n * block_size].reshape(n, block_size).mean(axis=1)

    energy_cut, _, _ = red.detect_equilibration_window(energy_block, method="min_sse")
    density_cut, _, _ = red.detect_equilibration_window(density_block, method="min_sse")

    energy_ps = float(time_ps[int(energy_cut) * block_size])
    density_ps = float(time_ps[int(density_cut) * block_size])
    cut_ps = max(energy_ps, density_ps, 250.0)

    if abs(energy_ps - density_ps) > 500.0:
        warn(
            "Energy and density equilibration cutoff times differ by more than 500 ps",
            stacklevel=2,
        )

    return cut_ps


def analyse_liquid(
    log_file_liq: Path,
    log_file_gas: Path,
    temperature: float,
    pressure: float,
    n_molecules: int,
    equil_time_ps: float | None,
    block_size: int,
) -> dict[str, tuple[float, float]]:
    """
    Analyse thermodynamic properties for one liquid simulation.

    Parameters
    ----------
    log_file_liq
        Path to the NPT liquid phase production log.
    log_file_gas
        Path to the NVT gas phase production log.
    temperature
        Simulation temperature in K.
    pressure
        Simulation pressure in bar.
    n_molecules
        Number of molecules in the liquid simulation cell.
    equil_time_ps
        Discard frames after this time, or estimate if None.
    block_size
        Number of logged samples in each statistical block.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mean and standard error for each thermodynamic observable.
    """
    time_series, time_units = read_property_from_log(
        log_file_liq,
        "t",
    )
    density_series, density_units = read_property_from_log(
        log_file_liq,
        "density",
    )
    volume_series, volume_units = read_property_from_log(
        log_file_liq,
        "volume",
    )
    pot_energy_series, pot_energy_units = read_property_from_log(
        log_file_liq,
        "Epot",
    )
    kin_energy_series, kin_energy_units = read_property_from_log(
        log_file_liq,
        "Ekin",
    )
    pot_energy_series_gas, pot_energy_units_gas = read_property_from_log(
        log_file_gas,
        "Epot",
    )

    if equil_time_ps is None:
        teq_ps = detect_equilibration_time(
            pot_energy_series, density_series, time_series, block_size=100
        )
    else:
        teq_ps = equil_time_ps

    teq = int(np.argwhere(time_series >= teq_ps)[0, 0])

    return {
        "density": density(
            density_series,
            block_size=block_size,
            teq=teq,
        ),
        "cp": heat_capacity_cp(
            pot_energy_series,
            kin_energy_series,
            volume_series,
            temperature=temperature,
            pressure=pressure,
            n_molecules=n_molecules,
            block_size=block_size,
            teq=teq,
        ),
        "evaporation_enthalpy": evaporation_enthalpy(
            pot_energy_series,
            pot_energy_series_gas,
            volume_series,
            temperature=temperature,
            pressure=pressure,
            n_molecules=n_molecules,
            block_size=block_size,
            teq=teq,
        ),
        "compressibility": isothermal_compressibility(
            volume_series,
            temperature=temperature,
            block_size=block_size,
            teq=teq,
        ),
        "alpha": thermal_expansion(
            pot_energy_series,
            kin_energy_series,
            volume_series,
            temperature=temperature,
            pressure=pressure,
            block_size=block_size,
            teq=teq,
        ),
    }


def get_thermodynamic_property_labels(data_path):
    """
    Get labels for thermodynamic property systems.

    Parameters
    ----------
    data_path
        Path to the thermodynamic properties input data.

    Returns
    -------
    list[str]
        Labels identifying the available thermodynamic property systems.
    """
    return sorted(
        path.stem
        for path in (data_path / "equilibrated_structures_xyz").glob("*-liq.xyz")
    )


def analyse_thermodynamic_properties(
    *,
    models,
    info,
    calc_path,
    out_path,
    block_size,
    equil_time_ps,
):
    """
    Analyse thermodynamic properties for all selected models and systems.

    Parameters
    ----------
    models
        Models to analyse.
    info
        Structure information for the benchmark systems.
    calc_path
        Path to the calculation outputs.
    out_path
        Path where analysed structures and results are written.
    block_size
        Size of blocks used for statistical error estimation.
    equil_time_ps
        Discard frames after this time, or estimate if None.

    Returns
    -------
    dict[str, dict]
        Analysed thermodynamic properties, reference values, and statistical
        uncertainties for each model.
    """
    results = {
        property_name: {
            "ref": [],
            **{
                model_name: {
                    "label": [],
                    "value": [],
                    "stderr": [],
                }
                for model_name in models
            },
        }
        for property_name in PROPERTIES
    }

    ref_stored = False

    model_iterator = models

    if tqdm is not None:
        model_iterator = tqdm(models, desc="Models", unit="model", position=0)

    for model_name in model_iterator:
        model_path = calc_path / model_name
        labels = sorted(path.stem for path in model_path.glob("*-liq.xyz"))

        label_iterator = labels

        if tqdm is not None:
            label_iterator = tqdm(
                labels, desc=f"{model_name}", unit="system", position=1, leave=False
            )

        for label in label_iterator:
            label_gas = label.removesuffix("-liq") + "-gas"

            xyz_file = calc_path / model_name / f"{label}.xyz"
            xyz_gas_file = calc_path / model_name / f"{label_gas}.xyz"
            log_file = calc_path / model_name / f"{label}.log"
            log_file_gas = calc_path / model_name / f"{label_gas}.log"
            required_files = (
                xyz_file,
                xyz_gas_file,
                log_file,
                log_file_gas,
            )
            missing = [path.name for path in required_files if not path.exists()]
            if missing:
                warn(
                    f"Skipping {model_name}, {label}: missing {', '.join(missing)}",
                    stacklevel=2,
                )
                continue

            atoms = read(xyz_file)
            atoms_gas = read(xyz_gas_file)

            calculated = analyse_liquid(
                log_file_liq=log_file,
                log_file_gas=log_file_gas,
                temperature=atoms.info["exp_temperature"],
                pressure=atoms.info["exp_pressure"],
                n_molecules=atoms.info["n_molecules"],
                block_size=block_size,
                equil_time_ps=equil_time_ps,
            )

            for property_name, (value, stderr) in calculated.items():
                results[property_name][model_name]["label"].append(label)
                results[property_name][model_name]["value"].append(value)
                results[property_name][model_name]["stderr"].append(stderr)

                if not ref_stored:
                    reference_key = PROPERTIES[property_name]["reference"]
                    results[property_name]["ref"].append(atoms.info[reference_key])

            structs_dir = out_path / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)
            write(structs_dir / f"{label_gas}.xyz", atoms_gas)

        ref_stored = True

    return results


def property_fixture_factory(property_name: str, info, out_path, models):
    """
    Create a parity-plot fixture for a thermodynamic property.

    Parameters
    ----------
    property_name
        Name of the thermodynamic property.
    info
        Structure information for the benchmark systems.
    out_path
        Path where the parity plot data are written.
    models
        Models included in the benchmark.

    Returns
    -------
    pytest.FixtureDef
        Pytest fixture generating parity-plot results for the property.
    """
    prop = PROPERTIES[property_name]

    @pytest.fixture(name=f"{property_name}_results")
    @plot_parity(
        filename=out_path / f"figure_{property_name}.json",
        title=prop["title"],
        x_label=f"Predicted {prop['title']} / ({prop['unit']})",
        y_label=f"Reference {prop['title']} / ({prop['unit']})",
        hoverdata={"Labels": info["filenames"]},
        use_plotly_autorange=False,
    )
    def _results(thermodynamic_properties) -> dict[str, list]:
        """
        Return parity-plot results for one thermodynamic property.

        Parameters
        ----------
        thermodynamic_properties
            Analysed thermodynamic properties for all models.

        Returns
        -------
        dict[str, list]
            Reference values and predicted values for each model.
        """
        return {
            "ref": thermodynamic_properties[property_name]["ref"],
            **{
                model_name: thermodynamic_properties[property_name][model_name]["value"]
                for model_name in models
            },
        }

    return _results


def get_metrics_factory(models):
    """
    Create the thermodynamic properties metrics fixture.

    Parameters
    ----------
    models
        Models included in the benchmark.

    Returns
    -------
    Callable
        Pytest fixture computing thermodynamic property metrics.
    """

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
        return _get_metrics(models, thermodynamic_properties)

    return get_metrics


def _get_metrics(
    models,
    thermodynamic_properties,
) -> dict[str, dict[str, float]]:
    """
    Compute thermodynamic benchmark metrics.

    Parameters
    ----------
    models
        The models to get the metrics of.

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

        for model_name in models:
            pred = np.asarray(
                thermodynamic_properties[property_name][model_name]["value"]
            )
            stderr = np.asarray(
                thermodynamic_properties[property_name][model_name]["stderr"]
            )
            mask = np.isfinite(ref)
            if not np.any(mask):
                results.setdefault(
                    f"{property_name}_MAE",
                    {},
                )[model_name] = np.nan
                results.setdefault(
                    f"{property_name}_MAZE",
                    {},
                )[model_name] = np.nan
                continue

            results.setdefault(
                f"{property_name}_MAE",
                {},
            )[model_name] = mae(
                ref[mask],
                pred[mask],
            )

            results.setdefault(
                f"{property_name}_MAZE",
                {},
            )[model_name] = maze(
                ref[mask],
                pred[mask],
                stderr[mask],
            )

    return results


def build_table_factory(out_path, default_tooltips, default_thresholds, d3_model_names):
    """
    Create the thermodynamic properties metrics table fixture.

    Parameters
    ----------
    out_path
        Path where the metrics table is written.
    default_tooltips
        Tooltips for the benchmark metrics.
    default_thresholds
        Thresholds for the benchmark metrics.
    d3_model_names
        Mapping of model names including dispersion corrections.

    Returns
    -------
    Callable
        Pytest fixture generating the metrics table.
    """

    @pytest.fixture
    @build_table(
        filename=out_path / "thermodynamic_properties_metrics_table.json",
        metric_tooltips=default_tooltips,
        thresholds=default_thresholds,
        mlip_name_map=d3_model_names,
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

    return metrics


def get_struct_info_thermodynamic_properties(calc_path, out_path):
    """
    Get structure information for the thermodynamic properties benchmark.

    Parameters
    ----------
    calc_path
        Path to the calculation outputs.
    out_path
        Path where structure information is written.

    Returns
    -------
    dict
        Structure information for the benchmark systems.
    """
    return get_struct_info(
        calc_path=calc_path,
        glob_pattern="*-liq.xyz",
        index=0,
        write_info=True,
        write_structs=True,
        out_path=out_path,
        include_filenames=True,
    )


def thermodynamic_properties_factory(models, info, calc_path, out_path):
    """
    Create the thermodynamic properties analysis fixture.

    Parameters
    ----------
    models
        Models included in the benchmark.
    info
        Structure information for the benchmark systems.
    calc_path
        Path to the calculation outputs.
    out_path
        Path where analysed results are written.

    Returns
    -------
    Callable
        Pytest fixture analysing thermodynamic properties.
    """

    @pytest.fixture
    def thermodynamic_properties(
        block_size: int,
        equil_time_ps: float,
    ) -> dict[str, dict]:
        """
        Analyse thermodynamic properties for all systems and models.

        Parameters
        ----------
        block_size
            The size of blocks used for error estimate.
        equil_time_ps
            Discard frames after this time, or estimate if None.

        Returns
        -------
        dict[str, dict]
            Reference values, predictions, and statistical uncertainties.
        """
        return analyse_thermodynamic_properties(
            models=models,
            info=info,
            calc_path=calc_path,
            out_path=out_path,
            block_size=block_size,
            equil_time_ps=equil_time_ps,
        )

    return thermodynamic_properties


def get_property_results_factory(models):
    """
    Create a thermodynamic property result extraction function.

    Parameters
    ----------
    models
        Models included in the benchmark.

    Returns
    -------
    Callable
        Function extracting parity-plot results for a property.
    """

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
                for model_name in models
            },
        }


def detailed_results_factory(models, out_path):
    """
    Create the detailed thermodynamic results fixture.

    Parameters
    ----------
    models
        Models included in the benchmark.
    out_path
        Path where detailed result files are written.

    Returns
    -------
    Callable
        Pytest fixture writing detailed per-CAS results.
    """

    @pytest.fixture
    def detailed_results_output(
        thermodynamic_properties,
        detailed_results,
    ):
        """
        Write detailed per-CAS thermodynamic property results.

        Parameters
        ----------
        thermodynamic_properties
            Analysed thermodynamic properties for all models.
        detailed_results
            Whether to write detailed per-CAS results.
        """
        if not detailed_results:
            return

        for model_name in models:
            results = {}

            for property_name in PROPERTIES:
                model_results = thermodynamic_properties[property_name][model_name]

                labels = model_results["label"]
                values = model_results["value"]
                errors = model_results["stderr"]

                for label, value, error in zip(labels, values, errors, strict=True):
                    cas = label.removesuffix("-liq")

                    results.setdefault(cas, {})
                    results[cas][property_name] = float(value)
                    results[cas][f"{property_name}_err"] = float(error)

            model_dir = out_path / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            with open(
                model_dir / "detailed_results.json",
                "w",
                encoding="utf8",
            ) as f:
                json.dump(results, f, indent=2)

    return detailed_results_output
