"""Run Grambow organics NEB convergence benchmark app."""

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

BENCHMARK_NAME = "GrambowOrganics"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#grambow-organics"
)
DATA_PATH = APP_ROOT / "data" / "nebs" / "grambow_organics"


class GrambowOrganicsApp(BaseApp):
    """Grambow organics NEB benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        plots = {}
        for model in get_model_names(current_models):
            model_path = DATA_PATH / model
            if not (model_path / "figure_neb_overview.json").exists():
                continue
            graph_id = f"{BENCHMARK_NAME}-{model}-neb"
            plot = read_plot(model_path / "figure_neb_overview.json", id=graph_id)
            plots[model] = {
                "Convergence Rate": plot,
                "Score": plot,
            }
            profile_id = plot_from_scatter_files(
                scatter_id=graph_id,
                plot_id=f"{BENCHMARK_NAME}-profile",
                plot_dir=model_path / "profiles",
                clear_plot_ids=[f"{BENCHMARK_NAME}-struct"],
            )
            structs = {
                path.stem.removeprefix("band_"): (
                    f"/assets/nebs/grambow_organics/{model}/{path.name}"
                )
                for path in model_path.glob("band_*.extxyz")
            }
            struct_from_scatter(
                scatter_id=profile_id,
                struct_id=f"{BENCHMARK_NAME}-struct",
                structs=structs,
                mode="traj",
                follow_frames=True,
            )
        clear_from_table_cell(
            self.table_id,
            [f"{BENCHMARK_NAME}-profile", f"{BENCHMARK_NAME}-struct"],
        )
        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure",
            cell_to_plot=plots,
        )


def get_app() -> GrambowOrganicsApp:
    """
    Get Grambow organics benchmark app layout and callback registration.

    Returns
    -------
    GrambowOrganicsApp
        Benchmark layout and callback registration.
    """
    return GrambowOrganicsApp(
        name="Grambow Organics",
        framework_ids="mlip_audit",
        description=(
            "Fraction of nudged elastic band simulations that converge for 100 "
            "elementary organic reactions sampled from the Grambow dataset. "
            "Assesses whether the model can produce a stable reaction path."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "grambow_organics_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure"),
            Div(id=f"{BENCHMARK_NAME}-profile"),
            Div(id=f"{BENCHMARK_NAME}-struct"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8072, debug=True)
