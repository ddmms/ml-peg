"""Run GSCDB138 isomer app using shared helpers."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.gscdb138 import GSCDB138BenchmarkApp

BENCHMARK_NAME = "GSCDB138 Isomers"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/isomers.html#gscdb138"
DESCRIPTION = (
    "Performance in predicting relative energies for the 26 isomer and conformer "
    "datasets in the GSCDB138 collection, benchmarked against CCSD(T) references."
)
DATA_PATH = APP_ROOT / "data" / "isomers" / "GSCDB138_isomers"

DATASETS = [
    "ACONF",
    "AlkIsomer11",
    "Amino20x4",
    "BUT14DIOL",
    "C20C246",
    "C60ISO7",
    "DIE60",
    "EIE22",
    "ICONF",
    "IDISP",
    "ISO34",
    "ISOL23",
    "ISOMERIZATION20",
    "MCONF",
    "PArel",
    "PCONF21",
    "Pentane13",
    "SCONF",
    "Styrene42",
    "TAUT15",
    "UPU23",
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
