"""Run the YBCO lattice-parameters-vs-oxygen-content benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "YBCO lattice vs oxygen"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "bulk_crystal.html#ybco-lattice-parameters-vs-oxygen-content"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "YBCO_lattice_vs_O"
INFO_PATH = DATA_PATH / "info.json"


class YBCOLatticeApp(BaseApp):
    """YBCO lattice-vs-oxygen benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_ybco_lattice.json", id=f"{BENCHMARK_NAME}-figure"
        )

        # one structure per oxygen content, repeated for its a/b/c scatter points
        structs_dir = DATA_PATH / MODELS[0]
        structs = [
            f"/assets/bulk_crystal/YBCO_lattice_vs_O/{MODELS[0]}/{f.stem}.xyz"
            for f in sorted(structs_dir.glob("*.xyz"))
            for _ in range(3)
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE a (PBE)": scatter,
                "MAE b (PBE)": scatter,
                "MAE c (PBE)": scatter,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> YBCOLatticeApp:
    """
    Get YBCO lattice-vs-oxygen benchmark app layout and callback registration.

    Returns
    -------
    YBCOLatticeApp
        Benchmark layout and callback registration.
    """
    return YBCOLatticeApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance when predicting the YBa2Cu3O(6+x) lattice parameters a, b, c "
            "as a function of oxygen content (6.0-7.0), including the orthorhombic-to-"
            "tetragonal phase transition, against CP2K PBE."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ybco_lattice_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    ybco_lattice_app = get_app()
    full_app.layout = ybco_lattice_app.layout
    ybco_lattice_app.register_callbacks()
    full_app.run(port=8055, debug=True)
