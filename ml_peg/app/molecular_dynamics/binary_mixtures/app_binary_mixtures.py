"""Run binary mixtures app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    clear_from_table_cell,
    plot_from_scatter_files,
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CATEGORY = "molecular_dynamics"
BENCHMARK_NAME = "Binary mixtures"
DOCS_URL = (
    f"https://ddmms.github.io/ml-peg/user_guide/benchmarks/{CATEGORY}.html"
    "#binary-mixtures"
)
DATA_PATH = APP_ROOT / "data" / CATEGORY / "binary_mixtures"
INFO_PATH = DATA_PATH / "info.json"
ASSETS_DIR = f"/assets/{CATEGORY}/binary_mixtures"

# Figure shown for each table column, and the directory of per-point plots
# (named by box) opened by clicking a point in that figure
COLUMN_FIGURES = {
    "ΔHmix MAE": "parity_hmix",
    "ΔHmix MAZE": "parity_hmix",
    "Density MAE": "parity_density",
    "Density MAZE": "parity_density",
    "V^E MAE": "parity_vex",
    "V^E MAZE": "parity_vex",
    "Pure density MAE": "parity_pure_density",
    "Pure density MAZE": "parity_pure_density",
    "Flagged boxes": "flag_summary",
}
DETAIL_DIRS = {
    "parity_hmix": "hmix_curves",
    "parity_density": "convergence_density",
    "parity_vex": "vex_curves",
    "parity_pure_density": "convergence_density",
    "flag_summary": "convergence_h",
}
# Second panel, shown alongside the first: ΔHmix and V^E running averages
SECOND_DETAIL_DIRS = {
    "parity_hmix": "convergence_hmix",
    "parity_vex": "convergence_vex",
}


class BinaryMixturesApp(BaseApp):
    """Binary mixtures benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        figure_placeholder = f"{BENCHMARK_NAME}-figure-placeholder"
        detail_placeholder = f"{BENCHMARK_NAME}-detail-placeholder"
        second_detail_placeholder = f"{BENCHMARK_NAME}-second-detail-placeholder"
        struct_placeholder = f"{BENCHMARK_NAME}-struct-placeholder"

        cell_to_plot: dict[str, dict] = {}
        for model_name in MODELS:
            model_dir = DATA_PATH / model_name
            graphs = {
                figure: read_plot(
                    model_dir / f"{figure}.json",
                    id=f"{BENCHMARK_NAME}-{model_name}-{figure}",
                )
                for figure in DETAIL_DIRS
                if (model_dir / f"{figure}.json").exists()
            }
            if not graphs:
                continue
            cell_to_plot[model_name] = {
                column: graphs[figure]
                for column, figure in COLUMN_FIGURES.items()
                if figure in graphs
            }
            # Final frame of each box, keyed by the box label each point carries
            structs = {
                path.stem: f"{ASSETS_DIR}/{model_name}/{path.name}"
                for path in sorted(model_dir.glob("*.xyz"))
            }
            for figure in graphs:
                scatter_id = f"{BENCHMARK_NAME}-{model_name}-{figure}"
                detail_id = plot_from_scatter_files(
                    scatter_id=scatter_id,
                    plot_id=detail_placeholder,
                    plot_dir=model_dir / DETAIL_DIRS[figure],
                )
                sources = [scatter_id]
                if figure in SECOND_DETAIL_DIRS:
                    # Clicking a parity point or a point of its composition curve
                    # shows that composition's running average
                    sources.append(detail_id)
                    for source in sources:
                        plot_from_scatter_files(
                            scatter_id=source,
                            plot_id=second_detail_placeholder,
                            plot_dir=model_dir / SECOND_DETAIL_DIRS[figure],
                            graph_id=f"{scatter_id}-second-plot",
                        )
                if structs:
                    for source in sources:
                        struct_from_scatter(
                            scatter_id=source,
                            struct_id=struct_placeholder,
                            structs=structs,
                            follow_frames=True,
                        )

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=figure_placeholder,
            cell_to_plot=cell_to_plot,
        )
        clear_from_table_cell(
            table_id=self.table_id,
            plot_ids=[
                detail_placeholder,
                second_detail_placeholder,
                struct_placeholder,
            ],
        )


def get_app() -> BinaryMixturesApp:
    """
    Get binary mixtures benchmark app layout and callback registration.

    Returns
    -------
    BinaryMixturesApp
        Benchmark layout and callback registration.
    """
    return BinaryMixturesApp(
        name=BENCHMARK_NAME,
        description=(
            "Enthalpy of mixing, mixture density, excess molar volume and pure "
            "density of 10 binary liquid mixtures (30 compositions, 2 aqueous), "
            "against experiment, from 1 ns NPT MD after 0.2 ns equilibration. "
            "Click a cell for a parity plot, then a point for its composition "
            "curve or time series and final structure (ΔHmix and V^E points also "
            "show their running average; click a point of the composition curve "
            "to switch composition)."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "binary_mixtures_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-detail-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-second-detail-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    binary_mixtures_app = get_app()
    full_app.layout = binary_mixtures_app.layout
    binary_mixtures_app.register_callbacks()
    full_app.run(port=8074, debug=True)
