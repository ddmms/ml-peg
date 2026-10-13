"""Run Sellers torsion scan app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Sellers_Tors"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/conformers.html#sellers-tors"
)
DATA_PATH = APP_ROOT / "data" / "conformers" / "Sellers_Tors"
INFO_PATH = DATA_PATH / "info.json"


class SellersTorsApp(BaseApp):
    """Sellers torsion scan benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_sellers_tors.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        barrier_scatter = read_plot(
            DATA_PATH / "figure_sellers_tors_barriers.json",
            id=f"{BENCHMARK_NAME}-barrier-figure",
        )

        # Structure files are named {fragment}_{scan point}, in scatter-point order
        structs_dir = DATA_PATH / "structs"
        if structs_dir.exists():
            labels = sorted(
                (f.stem for f in structs_dir.glob("*.xyz")),
                key=lambda label: tuple(int(part) for part in label.split("_")),
            )
            structs = [
                f"/assets/conformers/Sellers_Tors/structs/{label}.xyz"
                for label in labels
            ]
        else:
            structs = []

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE": scatter,
                "RMSE": scatter,
                "Barrier MAE": barrier_scatter,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> SellersTorsApp:
    """
    Get Sellers torsion scan benchmark app layout and callback registration.

    Returns
    -------
    SellersTorsApp
        Benchmark layout and callback registration.
    """
    return SellersTorsApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting torsional energy profiles of 62 neutral "
            "drug-like fragments (alkyl, aryl, biaryl and aryl amide torsions) "
            "with 36 dihedral angles each. "
            "Reference data from CCSD(T)/CBS calculations on MP2/6-311+G**-optimised "
            "constrained geometries."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "sellers_tors_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
