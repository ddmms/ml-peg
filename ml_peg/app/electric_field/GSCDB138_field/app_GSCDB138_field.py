"""Run GSCDB138 electric field app using shared helpers."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.gscdb138 import GSCDB138BenchmarkApp

BENCHMARK_NAME = "GSCDB138 Electric Field"
CATEGORY = "electric_field"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/electric_field.html#gscdb138"
)
DESCRIPTION = (
    "Performance in predicting electric field properties for the 6 electric field "
    "datasets in the GSCDB138 collection, benchmarked against CCSD(T) references."
)
DATA_PATH = APP_ROOT / "data" / "electric_field" / "GSCDB138_field"

DATASETS = ["Dip146", "HR46", "OEEF", "Pol130", "T144", "V30"]


def get_app() -> GSCDB138BenchmarkApp:
    """
    Get GSCDB138 benchmark app layout and callback registration.

    Returns
    -------
    GSCDB138BenchmarkApp
        Benchmark layout and callback registration.
    """
    return GSCDB138BenchmarkApp(
        name=BENCHMARK_NAME,
        description=DESCRIPTION,
        docs_url=DOCS_URL,
        data_path=DATA_PATH,
        datasets=DATASETS,
    )


if __name__ == "__main__":
    from ml_peg.app import APP_ROOT

    full_app = Dash(__name__, assets_folder=(APP_ROOT / "data"))

    gscdb_app = get_app()
    full_app.layout = gscdb_app.layout
    gscdb_app.register_callbacks()

    full_app.run(port=8052, debug=True)
