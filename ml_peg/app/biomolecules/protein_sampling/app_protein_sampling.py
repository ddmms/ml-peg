"""Run protein sampling benchmark app."""

from __future__ import annotations

from dash import Input, Output, callback, ctx, dcc
from dash.html import Div, Iframe, Label

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html

BENCHMARK_NAME = "ProteinSampling"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/biomolecules.html#protein-sampling"
DATA_PATH = APP_ROOT / "data" / "biomolecules" / "protein_sampling"


PROTEIN_LABELS = {
    "chignolin_1uao_xray": "Chignolin",
    "trp_cage_2jof_xray": "Trp-cage",
    "villin_capped_solvated": "Capped villin headpiece",
}


class ProteinSamplingApp(BaseApp):
    """Protein sampling benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Show the residue breakdown for the selected metric."""
        plots = {}
        for metric in self.metrics:
            if not metric.startswith("Backbone "):
                continue
            slug = metric.lower().replace(" ", "_")
            plots[metric] = read_plot(
                DATA_PATH / f"figure_{slug}.json", id=f"{BENCHMARK_NAME}-plot"
            )
        plot_from_table_column(self.table_id, f"{BENCHMARK_NAME}-figure", plots)

        @callback(
            Output(f"{BENCHMARK_NAME}-structure-controls", "children"),
            Input(f"{BENCHMARK_NAME}-plot", "clickData"),
            Input(f"{BENCHMARK_NAME}-plot", "figure"),
        )
        def select_protein(click_data, _figure):
            """
            Offer starting structures that contributed to the clicked point.

            Parameters
            ----------
            click_data
                Plotly point selection with the contributing protein identifiers.
            _figure
                Current figure. Changing metrics clears the previous selection.

            Returns
            -------
            str or list
                Selection guidance or the protein selector and viewer container.
            """
            if (
                not click_data
                or f"{BENCHMARK_NAME}-plot.figure" in ctx.triggered_prop_ids
            ):
                return "Select a residue point to view its contributing proteins."
            point = click_data["points"][0]
            names = [
                name for name in point.get("customdata", []) if name in PROTEIN_LABELS
            ]
            if not names:
                return "No starting structure is available for this point."
            selector = f"{BENCHMARK_NAME}-protein"
            return [
                Label(f"Starting structure for {point['x']}", htmlFor=selector),
                dcc.Dropdown(
                    id=selector,
                    options=[{"label": PROTEIN_LABELS[n], "value": n} for n in names],
                    value=names[0],
                    clearable=False,
                ),
                Div(id=f"{BENCHMARK_NAME}-viewer"),
            ]

        @callback(
            Output(f"{BENCHMARK_NAME}-viewer", "children"),
            Input(f"{BENCHMARK_NAME}-protein", "value"),
        )
        def show_structure(name):
            """
            Display the selected solvated protein before dynamics.

            Parameters
            ----------
            name
                Identifier of the selected protein.

            Returns
            -------
            Iframe or None
                Structure viewer when the identifier matches a benchmark protein.
            """
            if name in PROTEIN_LABELS:
                return Iframe(
                    title=PROTEIN_LABELS[name],
                    srcDoc=generate_weas_html(
                        f"/assets/biomolecules/protein_sampling/{name}.xyz"
                    ),
                    style={"width": "100%", "height": "550px", "border": "none"},
                )
            return None


def get_app() -> ProteinSamplingApp:
    """
    Get protein sampling benchmark app layout and callback registration.

    Returns
    -------
    ProteinSamplingApp
        Benchmark layout and callback registration.
    """
    return ProteinSamplingApp(
        name="Protein Sampling",
        framework_ids="mlip_audit",
        description=(
            "Performance in exploring protein conformational space during molecular "
            "dynamics. Sampled backbone dihedral distributions are compared against "
            "reference distributions via distribution RMSD, Hellinger distance, and "
            "the ratio of outlying conformations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "protein_sampling_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure"),
            Div(id=f"{BENCHMARK_NAME}-structure-controls"),
        ],
    )
