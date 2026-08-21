"""Run thermodynamic properties app."""

# ml_peg/app/molecular_dynamics/thermodynamic_properties/
# app_thermodynamic_properties.py
from __future__ import annotations

from ml_peg.app import APP_ROOT
from ml_peg.app.molecular_dynamics.thermodynamic_properties.utils import (
    thermodynamic_properties_app_factory,
)

BENCHMARK_NAME = "Precomputed Thermodynamic Properties"

DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "thermodynamic_properties"

DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "molecular_dynamics.html#thermodynamic-properties"
)


def get_app():
    """
    Return the app via the factory.

    Returns
    -------
    BaseApp
        Configured thermodynamic properties benchmark app.
    """
    return thermodynamic_properties_app_factory(
        benchmark_name=BENCHMARK_NAME,
        data_path=DATA_PATH,
        benchmark_path="thermodynamic_properties",
        docs_url=DOCS_URL,
    )
