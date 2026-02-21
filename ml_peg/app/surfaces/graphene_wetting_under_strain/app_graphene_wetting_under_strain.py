"""Run graphene wetting under strain app."""

from __future__ import annotations

from dash import Dash, Input, Output, callback
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe
import yaml

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

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


def plot_from_table_cell_with_erasure(
    table_id, plot_id, plot_id_erase, struct_id_erase, cell_to_plot
):
    """
    Attach callback to show plot when a table cell is clicked.

    Parameters
    ----------
    table_id
        ID for Dash table being clicked.
    plot_id
        ID for Dash plot placeholder Div.
    plot_id_erase
        ID of plot to be erased.
    struct_id_erase
        ID of struct to be erased.
    cell_to_plot
        Nested dictionary of model names, column names, and plot to show.
    """

    @callback(
        Output(plot_id, "children", allow_duplicate=True),
        Output(plot_id_erase, "children", allow_duplicate=True),
        Output(struct_id_erase, "children", allow_duplicate=True),
        Input(table_id, "active_cell"),
        Input(table_id, "data"),
        prevent_initial_call="initial_duplicate",
    )
    def show_plot(active_cell, current_table_data):
        """
        Register callback to show plot when a table cell is clicked.

        Parameters
        ----------
        active_cell
            Clicked cell in Dash table.
        current_table_data
            Current table data (includes live updates from callbacks).

        Returns
        -------
        Div
            Message explaining interactivity, or plot on cell click.
        Div
            Message explaining interactivity.
        Div
            Message explaining interactivity.
        """
        if not active_cell:
            return (
                Div("Click on a metric to view plot."),
                Div("Click on a metric to view plot."),
                Div("Click on a metric to view the structure."),
            )
        column_id = active_cell.get("column_id", None)
        row_id = active_cell.get("row_id", None)
        row_index = active_cell.get("row", None)
        if current_table_data and row_index is not None:
            try:
                cell_value = current_table_data[row_index].get(column_id)
                if cell_value is None:
                    return (
                        Div("No data available for this model."),
                        Div("Click on a metric to view plot."),
                        Div("Click on a metric to view the structure."),
                    )
            except (IndexError, KeyError, TypeError):
                pass
        if row_id in cell_to_plot and column_id in cell_to_plot[row_id]:
            return (
                Div(cell_to_plot[row_id][column_id]),
                Div("Click on a metric to view plot."),
                Div("Click on a metric to view the structure."),
            )
        return (
            Div("Click on a metric to view plot."),
            Div("Click on a metric to view plot."),
            Div("Click on a metric to view the structure."),
        )


def plot_and_struct_from_scatter(scatter_id, plot_id, plots_list, struct_id, structs):
    """
    Attach callback to show a structure when a scatter point is clicked.

    Parameters
    ----------
    scatter_id
        ID for Dash scatter being clicked.
    plot_id
        ID for Dash plot placeholder Div where new plot will be rendered.
    plots_list
        List of plots to show, in same order as scatter data.
    struct_id
        ID for Dash plot placeholder Div where structures will be visualised.
    structs
        List of structure filenames in same order as scatter data to be visualised.
    """

    @callback(
        Output(plot_id, "children", allow_duplicate=True),
        Output(struct_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_plot_and_struct(click_data):
        """
        Register callback to show plot and structure when a scatter point is clicked.

        Parameters
        ----------
        click_data
            Clicked data point in scatter plot.

        Returns
        -------
        Div
            Plot on scatter click.
        Div
            Visualised structure on plot click.
        """
        if not click_data:
            return Div("Click on a metric to view plot."), Div(
                "Click on a metric to view the structure."
            )
        idx = click_data["points"][0]["pointNumber"]
        return Div(plots_list[idx]), Div(
            Iframe(
                srcDoc=generate_weas_html(structs[idx], "struct", 0),
                style={
                    "height": "550px",
                    "width": "100%",
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                },
            )
        )


def struct_from_scatter_custom(scatter_id, struct_id, structs):
    """
    Attach callback to show a structure when a scatter point is clicked.

    Parameters
    ----------
    scatter_id
        ID for Dash scatter being clicked.
    struct_id
        ID for Dash plot placeholder Div where structures will be visualised.
    structs
        List of structure filenames in same order as scatter data to be visualised.
    """

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_struct(click_data):
        """
        Register callback to show structure when a scatter point is clicked.

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
            raise PreventUpdate()
        idx = click_data["points"][0]["pointNumber"]
        return Div(
            Iframe(
                srcDoc=generate_weas_html(structs[idx], "struct", 0),
                style={
                    "height": "550px",
                    "width": "100%",
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                },
            )
        )


class GrapheneWettingUnderStrainApp(BaseApp):
    """Graphene wetting under strain benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        parity_plots = {
            model: {
                "All Adsorption Energies MAE": read_plot(
                    DATA_PATH / model / "figure_all_parity.json",
                    id=f"{BENCHMARK_NAME}-{model}-all-parity-figure",
                ),
                "Binding Energies MAE": read_plot(
                    DATA_PATH / model / "figure_binding_energies_parity.json",
                    id=f"{BENCHMARK_NAME}-{model}-binding-energies-parity-figure",
                ),
                "Binding Lengths MAE": read_plot(
                    DATA_PATH / model / "figure_binding_lengths_parity.json",
                    id=f"{BENCHMARK_NAME}-{model}-binding-lengths-parity-figure",
                ),
            }
            for model in MODELS
        }

        structs_from_all = {model: [] for model in MODELS}
        for model in MODELS:
            for orientation in ORIENTATIONS:
                for strain in STRAINS:
                    xyz_files = sorted(
                        (DATA_PATH / model / "structs").glob(
                            f"{orientation}_{strain}_L*.xyz"
                        )
                    )
                    for xyz_file in xyz_files:
                        structs_from_all[model].append(
                            f"assets/surfaces/graphene_wetting_under_strain/{model}/structs/{xyz_file.name}"
                        )

        binding_curve_plots = {model: [] for model in MODELS}
        for model in MODELS:
            for orientation in ORIENTATIONS:
                for strain in STRAINS:
                    binding_curve_plots[model].append(
                        read_plot(
                            DATA_PATH / model / f"figure_{orientation}_{strain}.json",
                            id=f"{BENCHMARK_NAME}-{model}-{orientation}-{strain[0:3]}",
                        )
                    )

        n_distances = len(
            list(
                (DATA_PATH / MODELS[0] / "structs").glob(
                    f"{ORIENTATIONS[0]}_{STRAINS[0]}_L*.xyz"
                )
            )
        )
        curve_plots_from_all = {model: [] for model in MODELS}
        for model in MODELS:
            for i in range(len(ORIENTATIONS)):
                for j in range(len(STRAINS)):
                    idx = (i * len(STRAINS)) + j
                    for _ in range(n_distances):
                        curve_plots_from_all[model].append(
                            binding_curve_plots[model][idx]
                        )

        structs_from_binding_curves = {model: {} for model in MODELS}
        for model in MODELS:
            structs_from_binding_curves[model] = {
                orientation: {} for orientation in ORIENTATIONS
            }
            for orientation in ORIENTATIONS:
                structs_from_binding_curves[model][orientation] = {
                    strain: [] for strain in STRAINS
                }
                for strain in STRAINS:
                    xyz_files = sorted(
                        (DATA_PATH / model / "structs").glob(
                            f"{orientation}_{strain}_L*.xyz"
                        )
                    )
                    for xyz_file in xyz_files:
                        structs_from_binding_curves[model][orientation][strain].append(
                            f"assets/surfaces/graphene_wetting_under_strain/{model}/structs/{xyz_file.name}"
                        )

        plot_from_table_cell_with_erasure(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            plot_id_erase=f"{BENCHMARK_NAME}-subfigure-placeholder",
            struct_id_erase=f"{BENCHMARK_NAME}-struct-placeholder",
            cell_to_plot=parity_plots,
        )

        for model in MODELS:
            plot_and_struct_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{model}-all-parity-figure",
                plot_id=f"{BENCHMARK_NAME}-subfigure-placeholder",
                plots_list=curve_plots_from_all[model],
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                structs=structs_from_all[model],
            )
            plot_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{model}-binding-energies-parity-figure",
                plot_id=f"{BENCHMARK_NAME}-subfigure-placeholder",
                plots_list=binding_curve_plots[model],
            )
            plot_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{model}-binding-lengths-parity-figure",
                plot_id=f"{BENCHMARK_NAME}-subfigure-placeholder",
                plots_list=binding_curve_plots[model],
            )
            for orientation in ORIENTATIONS:
                for strain in STRAINS:
                    struct_from_scatter_custom(
                        scatter_id=f"{BENCHMARK_NAME}-{model}-{orientation}-{strain[0:3]}",
                        struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                        structs=structs_from_binding_curves[model][orientation][strain],
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
            Div(id=f"{BENCHMARK_NAME}-subfigure-placeholder"),
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
