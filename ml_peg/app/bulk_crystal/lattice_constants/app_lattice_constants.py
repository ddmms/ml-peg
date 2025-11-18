"""Run lattice constants benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Lattice constants"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#lattice-constants"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "lattice_constants"


class LatticeConstantsApp(BaseApp):
    """Lattice constants benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_exp = read_plot(
            DATA_PATH / "figure_lattice_consts_exp.json", id=f"{BENCHMARK_NAME}-figure"
        )
        scatter_dft = read_plot(
            DATA_PATH / "figure_lattice_consts_dft.json", id=f"{BENCHMARK_NAME}-figure"
        )

        structs_dir = DATA_PATH / MODELS[0]
        structs = []
        for struct_file in sorted(structs_dir.glob("*.xyz")):
            if struct_file.stem == "SiC":
                structs.extend(
                    [
                        f"assets/bulk_crystal/lattice_constants/{MODELS[0]}/{struct_file.stem}.xyz"
                    ]
                    * 2
                )
            else:
                structs.append(
                    f"assets/bulk_crystal/lattice_constants/{MODELS[0]}/{struct_file.stem}.xyz"
                )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE (Experimental)": scatter_exp,
                "MAE (PBE)": scatter_dft,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> LatticeConstantsApp:
    """
    Get lattice constants benchmark app layout and callback registration.

    Returns
    -------
    LatticeConstantsApp
        Benchmark layout and callback registration.
    """
    return LatticeConstantsApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance when predicting lattice constants for 23 solids, including "
            "pure elements, binary compounds and semiconductors."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "lattice_constants_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    lattice_constants_app = get_app()
    full_app.layout = lattice_constants_app.layout
    lattice_constants_app.register_callbacks()
    full_app.run(port=8054, debug=True)
