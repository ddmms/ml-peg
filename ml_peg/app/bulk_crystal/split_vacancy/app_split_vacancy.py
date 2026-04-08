"""Run split vacancy benchmark app."""

from __future__ import annotations

from dash import Dash, Input, Output, callback
from dash.html import Div, Iframe

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Split vacancy"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#split-vacancy"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "split_vacancy"

# for dash, assets/ is equivalent to APP_ROOT/data/
STRUCTS_URL = "assets/bulk_crystal/split_vacancy"


def struct_pair_from_violin(
    violin_id: str,
    struct_id: str,
    functional: str,
) -> None:
    """
    Register callback to show ref and MLIP structures when a violin point is clicked.

    Parameters
    ----------
    violin_id
        ID for Dash violin plot being clicked.
    struct_id
        ID for Dash placeholder Div where structures will be visualised.
    functional
        DFT functional (``"pbe"`` or ``"pbesol"``), used to determine structure paths.
    """

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(violin_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_struct(click_data):
        """
        Register callback to show structures when point clicked. See build_callbacks.py.

        Parameters
        ----------
        click_data
            Clicked data point in scatter plot.

        Returns
        -------
        Div
            Visualised structure on plot click.
        """
        if not click_data:
            return Div("Click on a point to view structures.")

        point = click_data["points"][0]
        model_name = point["x"]
        mp_id = point["customdata"][0]
        formula = point["customdata"][1]
        cation = point["customdata"][2]
        vac_type = point["customdata"][3]
        frame_id = int(point["customdata"][4])

        xyz_name = "normal_vacancy.xyz" if vac_type == "NV" else "split_vacancy.xyz"
        material_dir = f"{formula}-{mp_id}"

        prefix = f"{STRUCTS_URL}/{functional}"
        ref_struct = f"{prefix}/ref/{material_dir}/{cation}/{xyz_name}"
        mlip_struct = f"{prefix}/{model_name}/{material_dir}/{cation}/{xyz_name}"

        ref_html = generate_weas_html(ref_struct, mode="traj", index=frame_id)
        mlip_html = generate_weas_html(mlip_struct, mode="traj", index=frame_id)

        iframe_style = {
            "height": "550px",
            "width": "100%",
            "border": "1px solid #ddd",
            "borderRadius": "5px",
        }
        return Div(
            [
                Div(
                    [Iframe(srcDoc=ref_html, style=iframe_style)],
                    style={"width": "50%", "paddingRight": "4px"},
                ),
                Div(
                    [Iframe(srcDoc=mlip_html, style=iframe_style)],
                    style={"width": "50%", "paddingLeft": "4px"},
                ),
            ],
            style={"display": "flex"},
        )


class SplitVacancyApp(BaseApp):
    """Split vacancy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_pbesol = read_plot(
            DATA_PATH / "figure_formation_energies_pbesol.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        scatter_pbe = read_plot(
            DATA_PATH / "figure_formation_energies_pbe.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        max_dist_violin_pbesol = read_plot(
            DATA_PATH / "figure_max_dist_pbesol.json",
            id=f"{BENCHMARK_NAME}-figure-pbesol",
        )
        max_dist_violin_pbe = read_plot(
            DATA_PATH / "figure_max_dist_pbe.json",
            id=f"{BENCHMARK_NAME}-figure-pbe",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE (PBEsol)": scatter_pbesol,
                "Spearman's (PBEsol)": scatter_pbesol,
                "Max Dist (PBEsol)": max_dist_violin_pbesol,
                "MAE (PBE)": scatter_pbe,
                "Spearman's (PBE)": scatter_pbe,
                "Max Dist (PBE)": max_dist_violin_pbe,
            },
        )

        struct_pair_from_violin(
            violin_id=f"{BENCHMARK_NAME}-figure-pbesol",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            functional="pbesol",
        )
        struct_pair_from_violin(
            violin_id=f"{BENCHMARK_NAME}-figure-pbe",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            functional="pbe",
        )


def get_app() -> SplitVacancyApp:
    """
    Get split vacancy benchmark app layout and callback registration.

    Returns
    -------
    SplitVacancyApp
        Benchmark layout and callback registration.
    """
    return SplitVacancyApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance predicting the formation energy of split "
            "vacancies from fully ionised vacancies in nitrides and oxides."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "split_vacancy_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    split_vacancy_app = get_app()
    full_app.layout = split_vacancy_app.layout
    split_vacancy_app.register_callbacks()
    full_app.run(port=8054, debug=True)
