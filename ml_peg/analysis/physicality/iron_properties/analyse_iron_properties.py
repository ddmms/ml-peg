"""Analyse BCC iron properties benchmark.

This analysis combines EOS, elastic, Bain path, defect, surface, stacking fault,
dislocation, and fracture properties.

Reference
---------
Zhang, L., Csányi, G., van der Giessen, E., & Maresca, F. (2023).
Efficiency, Accuracy, and Transferability of Machine Learning Potentials:
Application to Dislocations and Cracks in Iron.
arXiv:2307.10072. https://arxiv.org/abs/2307.10072
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "iron_properties" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "iron_properties"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

# DFT reference values
DFT_REFERENCE = {
    # EOS properties
    'a0': 2.831,           # Lattice parameter (Å)
    'B0': 178.0,           # Bulk modulus (GPa)
    'E_bcc_fcc': 83.5,     # BCC-FCC energy difference (meV/atom)
    # Defect properties
    'E_vac': 2.02,         # Vacancy formation energy (eV)
    'gamma_100': 2.41,     # Surface energy (J/m²)
    'gamma_110': 2.37,
    'gamma_111': 2.58,
    'gamma_112': 2.48,
    'gamma_us_110': 0.75,  # Unstable SFE (J/m²)
    'gamma_us_112': 1.12,
    # Dislocation properties (approximate)
    'core_energy_screw_111': 1.8,     # eV
    'core_energy_edge_111_110': 2.2,  # eV
    # Crack K_Griffith (MPa*sqrt(m))
    'K_Griffith_1': 1.05,
    'K_Griffith_2': 1.02,
    'K_Griffith_3': 0.98,
    'K_Griffith_4': 0.95,
}

# Dislocation type mapping
DISLOCATION_NAMES = {
    'edge_100_010': 'Edge a0[100](010)',
    'edge_100_011': 'Edge a0[100](011)',
    'edge_111_110': 'Edge a0/2[111](110)',
    'mixed_111': 'Mixed 70.5° a0/2[111](110)',
    'screw_111': 'Screw a0/2[111](112)',
}


def load_model_results(model_name: str) -> dict[str, Any] | None:
    """Load iron properties results for a model."""
    json_path = CALC_PATH / model_name / "results.json"
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


def load_eos_curve(model_name: str) -> pd.DataFrame:
    """Load EOS curve data for a model."""
    csv_path = CALC_PATH / model_name / "eos_curve.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_bain_curve(model_name: str) -> pd.DataFrame:
    """Load Bain path curve data for a model."""
    csv_path = CALC_PATH / model_name / "bain_path.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_sfe_110_curve(model_name: str) -> pd.DataFrame:
    """Load SFE 110 curve data for a model."""
    csv_path = CALC_PATH / model_name / "sfe_110_curve.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_sfe_112_curve(model_name: str) -> pd.DataFrame:
    """Load SFE 112 curve data for a model."""
    csv_path = CALC_PATH / model_name / "sfe_112_curve.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_crack_ke_curve(model_name: str, crack_system: int) -> pd.DataFrame:
    """Load crack K-E curve data for a model."""
    csv_path = CALC_PATH / model_name / f"crack_{crack_system}_KE.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def compute_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Compute metrics from model results."""
    metrics: dict[str, float] = {}
    
    # ==========================================================================
    # EOS metrics
    # ==========================================================================
    eos = results.get('eos', {})
    if 'a0' in eos:
        a0_mlip = eos['a0']
        a0_error = abs(a0_mlip - DFT_REFERENCE['a0']) / DFT_REFERENCE['a0'] * 100
        metrics['a0 error (%)'] = a0_error
    
    if 'B0' in eos:
        B0_mlip = eos['B0']
        B0_error = abs(B0_mlip - DFT_REFERENCE['B0']) / DFT_REFERENCE['B0'] * 100
        metrics['B0 error (%)'] = B0_error
    
    # ==========================================================================
    # Bain path metrics
    # ==========================================================================
    bain = results.get('bain_path', {})
    if 'delta_E_meV' in bain:
        E_bcc_fcc_mlip = bain['delta_E_meV']
        E_bcc_fcc_error = abs(E_bcc_fcc_mlip - DFT_REFERENCE['E_bcc_fcc'])
        metrics['BCC-FCC ΔE error (meV)'] = E_bcc_fcc_error
    
    # ==========================================================================
    # Elastic constants metrics
    # ==========================================================================
    elastic = results.get('elastic', {})
    if 'C11' in elastic:
        metrics['C11 (GPa)'] = elastic['C11']
    if 'C12' in elastic:
        metrics['C12 (GPa)'] = elastic['C12']
    if 'C44' in elastic:
        metrics['C44 (GPa)'] = elastic['C44']
    
    # ==========================================================================
    # Vacancy metrics
    # ==========================================================================
    vacancy = results.get('vacancy', {})
    if 'E_vac' in vacancy:
        E_vac_mlip = vacancy['E_vac']
        E_vac_error = abs(E_vac_mlip - DFT_REFERENCE['E_vac']) / DFT_REFERENCE['E_vac'] * 100
        metrics['E_vac error (%)'] = E_vac_error
    
    # ==========================================================================
    # Surface energy metrics
    # ==========================================================================
    surfaces = results.get('surfaces', {})
    surface_errors = []
    
    for surface in ['100', '110', '111', '112']:
        key_mlip = f'gamma_{surface}'
        if key_mlip in surfaces:
            gamma_mlip = surfaces[key_mlip]
            gamma_dft = DFT_REFERENCE[key_mlip]
            error = abs(gamma_mlip - gamma_dft)
            surface_errors.append(error)
    
    if surface_errors:
        metrics['Surface MAE (J/m²)'] = np.mean(surface_errors)
    
    # ==========================================================================
    # Stacking fault metrics
    # ==========================================================================
    sfe_110 = results.get('sfe_110', {})
    if 'max_sfe' in sfe_110:
        max_sfe_110_mlip = sfe_110['max_sfe']
        max_sfe_110_error = abs(max_sfe_110_mlip - DFT_REFERENCE['gamma_us_110']) / DFT_REFERENCE['gamma_us_110'] * 100
        metrics['Max SFE 110 error (%)'] = max_sfe_110_error
    
    sfe_112 = results.get('sfe_112', {})
    if 'max_sfe' in sfe_112:
        max_sfe_112_mlip = sfe_112['max_sfe']
        max_sfe_112_error = abs(max_sfe_112_mlip - DFT_REFERENCE['gamma_us_112']) / DFT_REFERENCE['gamma_us_112'] * 100
        metrics['Max SFE 112 error (%)'] = max_sfe_112_error
    
    # ==========================================================================
    # Dislocation core energy metrics
    # ==========================================================================
    dislocations = results.get('dislocations', {})
    core_energies = []
    
    for disl_type, disl_data in dislocations.items():
        if isinstance(disl_data, dict) and 'core_energy' in disl_data:
            core_energies.append(disl_data['core_energy'])
            metrics[f'Core E {DISLOCATION_NAMES.get(disl_type, disl_type)} (eV)'] = disl_data['core_energy']
    
    if core_energies:
        metrics['Mean core energy (eV)'] = np.mean(core_energies)
    
    # ==========================================================================
    # Crack K-test metrics
    # ==========================================================================
    cracks = results.get('cracks', {})
    K_Griffith_values = []
    
    for crack_sys, crack_data in cracks.items():
        if isinstance(crack_data, dict) and 'K_Griffith' in crack_data:
            K_G = crack_data['K_Griffith']
            K_Griffith_values.append(K_G)
            metrics[f'K_Griffith {crack_data.get("name", f"System {crack_sys}")} (MPa√m)'] = K_G
    
    if K_Griffith_values:
        metrics['Mean K_Griffith (MPa√m)'] = np.mean(K_Griffith_values)
    
    return metrics


def _load_all_results() -> dict[str, dict[str, Any]]:
    """Load results for all models."""
    all_results: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        results = load_model_results(model_name)
        if results is not None:
            all_results[model_name] = results
    return all_results


@pytest.fixture
def iron_eos_curves() -> dict[str, pd.DataFrame]:
    """Load EOS curves for all models."""
    curves: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        curve = load_eos_curve(model_name)
        if not curve.empty:
            curves[model_name] = curve
    return curves


@pytest.fixture
def iron_bain_curves() -> dict[str, pd.DataFrame]:
    """Load Bain path curves for all models."""
    curves: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        curve = load_bain_curve(model_name)
        if not curve.empty:
            curves[model_name] = curve
    return curves


@pytest.fixture
def iron_sfe_110_curves() -> dict[str, pd.DataFrame]:
    """Load SFE 110 curves for all models."""
    curves: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        curve = load_sfe_110_curve(model_name)
        if not curve.empty:
            curves[model_name] = curve
    return curves


@pytest.fixture
def iron_sfe_112_curves() -> dict[str, pd.DataFrame]:
    """Load SFE 112 curves for all models."""
    curves: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        curve = load_sfe_112_curve(model_name)
        if not curve.empty:
            curves[model_name] = curve
    return curves


@pytest.fixture
def iron_crack_curves() -> dict[str, dict[int, pd.DataFrame]]:
    """Load crack K-E curves for all models."""
    curves: dict[str, dict[int, pd.DataFrame]] = {}
    for model_name in MODELS:
        model_curves: dict[int, pd.DataFrame] = {}
        for crack_sys in [1, 2, 3, 4]:
            curve = load_crack_ke_curve(model_name, crack_sys)
            if not curve.empty:
                model_curves[crack_sys] = curve
        if model_curves:
            curves[model_name] = model_curves
    return curves


def collect_metrics() -> pd.DataFrame:
    """Gather metrics for all models."""
    metrics_rows: list[dict[str, float | str]] = []
    
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    
    all_results = _load_all_results()
    
    for model_name, results in all_results.items():
        model_metrics = compute_metrics(results)
        row = {"Model": model_name} | model_metrics
        metrics_rows.append(row)
    
    columns = ["Model"] + list(DEFAULT_THRESHOLDS.keys())
    
    return pd.DataFrame(metrics_rows).reindex(columns=columns)


@pytest.fixture
def iron_properties_collection() -> pd.DataFrame:
    """Collect iron properties metrics across all models."""
    return collect_metrics()


@pytest.fixture
def iron_properties_metrics_dataframe(
    iron_properties_collection: pd.DataFrame,
) -> pd.DataFrame:
    """Provide the aggregated iron properties metrics dataframe."""
    return iron_properties_collection


@pytest.fixture
@build_table(
    filename=OUT_PATH / "iron_properties_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=None,
)
def metrics(
    iron_properties_metrics_dataframe: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute iron properties metrics for all models.
    
    Parameters
    ----------
    iron_properties_metrics_dataframe
        Aggregated per-model metrics.
    
    Returns
    -------
    dict[str, dict]
        Mapping of metric names to per-model results.
    """
    metrics_df = iron_properties_metrics_dataframe
    metrics_dict: dict[str, dict[str, float | None]] = {}
    for column in metrics_df.columns:
        if column == "Model":
            continue
        values = [
            value if pd.notna(value) else None for value in metrics_df[column].tolist()
        ]
        metrics_dict[column] = dict(zip(metrics_df["Model"], values, strict=False))
    return metrics_dict


def test_iron_properties(metrics: dict[str, dict]) -> None:
    """Run iron properties analysis."""
    return
