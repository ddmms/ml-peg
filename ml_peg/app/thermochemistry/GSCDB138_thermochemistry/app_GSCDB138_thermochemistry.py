"""Run GSCDB138 thermochemistry app using shared helpers."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.gscdb138 import GSCDB138BenchmarkApp

BENCHMARK_NAME = "GSCDB138 Thermochemistry"
CATEGORY = "thermochemistry"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/thermochemistry.html#gscdb138"
)
DESCRIPTION = (
    "Performance in predicting relative energies for the 26 isomer and conformer "
    "datasets in the GSCDB138 collection, benchmarked against CCSD(T) references."
)
DATA_PATH = APP_ROOT / "data" / "thermochemistry" / "GSCDB138_thermochemistry"

DATASETS = [
    "AE11",
    "AE18",
    "AL2X6",
    "ALK8",
    "AlkAtom19",
    "ALKBDE10",
    "AlkIsod14",
    "BDE99MR",
    "BDE99nonMR",
    "BH76RC",
    "BSR36",
    "CR20",
    "DARC",
    "DC13",
    "DIPCS9",
    "EA50",
    "FH51",
    "G21EA",
    "G21IP",
    "G2RC24",
    "HAT707MR",
    "HAT707nonMR",
    "HEAVYSB11",
    "HNBrBDE18",
    "IP23",
    "IP30",
    "MB08-165",
    "MB16-43",
    "MX34",
    "NBPRC",
    "P34AE",
    "P34EA",
    "P34IP",
    "PA26",
    "PlatonicRE18",
    "PlatonicTAE6",
    "RC21",
    "RSE43",
    "SIE4x4",
    "SN13",
    "TAE_W4-17MR",
    "TAE_W4-17nonMR",
    "WCPT6",
    "YBDE18",
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
