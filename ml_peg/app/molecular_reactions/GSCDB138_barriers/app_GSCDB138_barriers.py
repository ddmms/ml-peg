"""Run GSCDB138 reaction barriers app using shared helpers."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.gscdb138 import GSCDB138BenchmarkApp

BENCHMARK_NAME = "GSCDB138 Molecular Reactions"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_reactions.html#gscdb138"
DESCRIPTION = (
    "Performance in predicting barrier heights for the 12 datasets in the GSCDB138"
    "collection, benchmarked against CCSD(T) references."
)
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "GSCDB138_barriers"

DATASETS = [
    "BH28",
    "BH46",
    "BH876",
    "BHDIV7",
    "BHPERI11",
    "BHROT27",
    "CRBH14",
    "DBH22",
    "INV23",
    "ORBH35",
    "PX9",
    "WCPT26",
]


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
