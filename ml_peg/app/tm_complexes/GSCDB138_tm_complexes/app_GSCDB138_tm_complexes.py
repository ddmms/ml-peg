"""Run GSCDB138 TM complex app using shared helpers."""

from __future__ import annotations

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.gscdb138 import GSCDB138BenchmarkApp

BENCHMARK_NAME = "GSCDB138 TM Complexes"
CATEGORY = "tm_complexes"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/tm_complexes.html#gscdb138"
)
DESCRIPTION = (
    "Performance in predicting transition metal complex energy "
    "datasets in the GSCDB138 collection, benchmarked against CCSD(T) references."
)
DATA_PATH = APP_ROOT / "data" / "tm_complexes" / "GSCDB138_tm_complexes"

DATASETS = [
    "3d4dIPSS",
    "CUAGAU83",
    "DAPD",
    "MME52",
    "MOBH28",
    "MOR13",
    "ROST61",
    "TMB11",
    "TMD10",
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
