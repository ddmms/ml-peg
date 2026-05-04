"""Run graphene wetting under strain app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div
import yaml

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_multi_scatters,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Graphene Wetting Under Strain"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#graphene-wetting-under-strain"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "graphene_wetting_under_strain"

CALC_PATH = CALCS_ROOT / "surfaces" / "graphene_wetting_under_strain" / "outputs"
with open(CALC_PATH / "database_info.yml") as fp:
    DATABASE_INFO = yaml.safe_load(fp)
ORIENTATIONS = DATABASE_INFO["orientations"]
STRAINS = DATABASE_INFO["strains"]


class GrapheneWettingUnderStrainApp(BaseApp):
    """Graphene wetting under strain benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        structs_from_curve = [
            f"/assets/surfaces/graphene_wetting_under_strain/structs/{orientation}_{strain}.xyz"
            for orientation in ORIENTATIONS
            for strain in STRAINS
        ] * (len(MODELS) + 1)

        structs_from_params = [
            [
                f"/assets/surfaces/graphene_wetting_under_strain/structs/{orientation}_{strain}.xyz"
                for strain in STRAINS
            ]
            for orientation in ORIENTATIONS
        ] * (len(MODELS) + 1)

        binding_curve_plot = read_plot(
            DATA_PATH / "figure_adsorption_energies.json",
            id=f"{BENCHMARK_NAME}-figure-adsorption-energies",
        )

        binding_energies_plot = read_plot(
            DATA_PATH / "figure_binding_energies.json",
            id=f"{BENCHMARK_NAME}-figure-binding-energies",
        )

        binding_lengths_plot = read_plot(
            DATA_PATH / "figure_binding_lengths.json",
            id=f"{BENCHMARK_NAME}-figure-binding-lengths",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "All Adsorption Energies MAE": binding_curve_plot,
                "Binding Energies MAE": binding_energies_plot,
                "Binding Lengths MAE": binding_lengths_plot,
            },
        )

        struct_from_multi_scatters(
            scatter_id=f"{BENCHMARK_NAME}-figure-adsorption-energies",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs_from_curve,
            mode="traj",
        )

        struct_from_multi_scatters(
            scatter_id=f"{BENCHMARK_NAME}-figure-binding-energies",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs_from_params,
            mode="traj",
        )

        struct_from_multi_scatters(
            scatter_id=f"{BENCHMARK_NAME}-figure-binding-lengths",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs_from_params,
            mode="traj",
        )


def get_app() -> GrapheneWettingUnderStrainApp:
    """
    Get graphene wetting under strain benchmark app layout and callback registration.

    Returns
    -------
    GrapheneWettingUnderStrainApp
        Benchmark layout and callback registration.
    """
    return GrapheneWettingUnderStrainApp(
        name=BENCHMARK_NAME,
        description=("Adsorption energies for water on graphene."),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "graphene_wetting_under_strain_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    graphene_wetting_under_strain_app = get_app()
    full_app.layout = graphene_wetting_under_strain_app.layout
    graphene_wetting_under_strain_app.register_callbacks()

    # Run app
    full_app.run(port=8052, debug=True)
