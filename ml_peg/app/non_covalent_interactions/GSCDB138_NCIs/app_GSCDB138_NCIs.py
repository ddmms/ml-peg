"""Run GSCDB138 non-covalent interactions app using shared helpers."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.gscdb138 import GSCDB138BenchmarkApp

BENCHMARK_NAME = "GSCDB138 Non-Covalent Interactions"
CATEGORY = "non_covalent_interactions"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/non_covalent_interactions.html#gscdb138"
DESCRIPTION = (
    "Performance in predicting relative energies for the non-covalent interaction "
    "datasets in the GSCDB138 collection, benchmarked against CCSD(T) references."
)
DATA_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "GSCDB138_NCIs"

DATASETS = [
    "3B-69",
    "3BHET",
    "A19Rel6",
    "A24",
    "ADIM6",
    "AHB21",
    "Bauza30",
    "BzDC215",
    "CARBHB8",
    "CHB6",
    "CT20",
    "DS14",
    "FmH2O10",
    "H2O16Rel4",
    "H2O20Rel9",
    "HB262",
    "HB49",
    "HCP32",
    "He3",
    "HEAVY28",
    "HSG",
    "HW30",
    "HW6Cl5",
    "HW6F",
    "IHB100",
    "IHB100x2",
    "IL16",
    "NBC10",
    "NC11",
    "O24",
    "O24x4",
    "PNICO23",
    "RG10N",
    "RG18",
    "S22",
    "S66",
    "S66Rel7",
    "Shields38",
    "SW49Bind22",
    "SW49Rel28",
    "TA13",
    "WATER27",
    "X40",
    "X40x5",
    "XB25",
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
