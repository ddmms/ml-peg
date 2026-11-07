"""Run MAD-BENCH app."""

from __future__ import annotations

from dash import Input, Output, callback, dcc
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate
from dash.html import Div, Label, P

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_with_download_controls,
    register_plot_settings_callbacks,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_density_plot_for_model, read_json
from ml_peg.app.utils.plot_helpers import INSTRUCTION_STYLE, TABLE_HINT
from ml_peg.app.utils.register_callbacks import register_plot_download_callbacks
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "MAD-BENCH"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#mad-bench"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "mad_bench"
ASSETS_PREFIX = "/assets/bulk_crystal/mad_bench"

REFERENCE_DROPDOWN_ID = f"{BENCHMARK_NAME}-reference-dropdown"
CELL_STORE_ID = f"{BENCHMARK_NAME}-cell-store"
PLOT_ID = f"{BENCHMARK_NAME}-figure-placeholder"
STRUCT_ID = f"{BENCHMARK_NAME}-struct-placeholder"
DIAGNOSTICS_NOTE_ID = f"{BENCHMARK_NAME}-diagnostics-note"
DIAGNOSTICS_TABLE_ID = f"{BENCHMARK_NAME}-diagnostics-table"

PROPERTY_LABELS = {"forces": "Force", "energy": "Energy", "stress": "Stress"}
PLOTTED_METRICS = {"Force MAE": "forces", "Stress MAE": "stress"}
DIAGNOSTIC_STATISTICS = (
    ("forces", "MAE"),
    ("forces", "RMSE"),
    ("forces", "p8"),
    ("forces", "sup"),
    ("stress", "MAE"),
    ("stress", "RMSE"),
    ("stress", "p8"),
    ("stress", "sup"),
)
OUTSIDE_DECLARED_COLUMN = "Outside declared training elements"


def figure_id(reference_key: str, model: str, name: str) -> str:
    """
    Get a Dash component ID for one parity plot.

    Parameters
    ----------
    reference_key
        Reference protocol key, which may contain dots.
    model
        Model name.
    name
        Property name, one of ``forces`` or ``stress``.

    Returns
    -------
    str
        Component ID without dots, which Dash does not allow.
    """
    return f"{BENCHMARK_NAME}-{reference_key.replace('.', '-')}-{model}-{name}-figure"


def format_statistic(value: float | None) -> str:
    """
    Format a diagnostic statistic, keeping unavailable values explicit.

    Parameters
    ----------
    value
        Statistic, or None if its population is incomplete.

    Returns
    -------
    str
        Value to four significant figures, or ``N/A``.
    """
    if value is None:
        return "N/A"

    return f"{value:.4g}"


def format_coverage(outcomes: dict[str, int]) -> str:
    """
    Format the evaluated structures out of the expected population.

    Parameters
    ----------
    outcomes
        Count of each prediction outcome.

    Returns
    -------
    str
        Structures with a valid prediction and structures expected.
    """
    return f"{outcomes.get('ok', 0)} / {sum(outcomes.values())}"


def format_failures(model_diagnostics: dict, properties: list[str]) -> str:
    """
    List the recorded failure statuses of every available property.

    Parameters
    ----------
    model_diagnostics
        Diagnostics of one model against one reference protocol.
    properties
        Properties labelled by the reference protocol.

    Returns
    -------
    str
        Property, status and count of each failure, or ``none``.
    """
    failures = []

    for name in properties:
        for status, count in sorted(model_diagnostics[name]["outcomes"].items()):
            if status != "ok":
                failures.append(f"{name} {status} {count}")

    return ", ".join(failures) or "none"


def format_outside_declared(outcomes: dict[str, int] | None) -> str:
    """
    Format structures with elements outside the declared training data.

    Parameters
    ----------
    outcomes
        Prediction outcomes of those structures, or None if nothing is declared.

    Returns
    -------
    str
        Number of such structures and how many of them failed.
    """
    if outcomes is None:
        return "not declared"

    n_total = sum(outcomes.values())
    n_failed = n_total - outcomes.get("ok", 0)

    return f"{n_total} ({n_failed} failed)"


def diagnostics_table_content(
    reference_diagnostics: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Build columns and rows of the unscored table for one reference protocol.

    Parameters
    ----------
    reference_diagnostics
        Diagnostics of one reference protocol written by the analysis.

    Returns
    -------
    tuple[list[dict], list[dict]]
        Table columns and one row per model, for the properties the protocol labels.
    """
    units = reference_diagnostics["units"]
    properties = [
        name
        for name in PROPERTY_LABELS
        if reference_diagnostics["n_structures"][name] > 0
    ]

    statistic_columns = {
        (name, statistic): f"{PROPERTY_LABELS[name]} {statistic} / {units[name]}"
        for name, statistic in DIAGNOSTIC_STATISTICS
        if name in properties
    }
    names = (
        ["MLIP"]
        + [f"{PROPERTY_LABELS[name]} evaluated / expected" for name in properties]
        + list(statistic_columns.values())
        + ["Failures", OUTSIDE_DECLARED_COLUMN]
    )

    rows = []

    for model_name in MODELS:
        if model_name not in reference_diagnostics["models"]:
            continue

        model_diagnostics = reference_diagnostics["models"][model_name]
        row = {"MLIP": model_name}

        for name in properties:
            row[f"{PROPERTY_LABELS[name]} evaluated / expected"] = format_coverage(
                model_diagnostics[name]["outcomes"]
            )

        for (name, statistic), column in statistic_columns.items():
            row[column] = format_statistic(model_diagnostics[name][statistic])

        row["Failures"] = format_failures(model_diagnostics, properties)
        row[OUTSIDE_DECLARED_COLUMN] = format_outside_declared(
            model_diagnostics["forces"]["outside_declared_elements"]
        )

        rows.append(row)

    return [{"name": name, "id": name} for name in names], rows


def diagnostics_note(reference_diagnostics: dict) -> str:
    """
    Describe the unscored table for one reference protocol.

    Parameters
    ----------
    reference_diagnostics
        Diagnostics of one reference protocol written by the analysis.

    Returns
    -------
    str
        Reference protocol, populations and the energy limitation.
    """
    n_structures = reference_diagnostics["n_structures"]

    note = (
        f"Unscored diagnostics against {reference_diagnostics['label']}. Each "
        "statistic is N/A unless every structure in its population has a valid "
        f"prediction. Forces cover all {n_structures['forces']:,} structures"
    )

    if n_structures["stress"]:
        note += f" and stress the {n_structures['stress']:,} with a reference stress"

    return (
        f"{note}. Energy errors are not reported until a reference energy "
        "convention is agreed."
    )


def build_reference_selector(diagnostics: dict) -> Div:
    """
    Build the fixed structure set and the reference protocol selector.

    Parameters
    ----------
    diagnostics
        Contents of the diagnostics file written by the analysis.

    Returns
    -------
    Div
        Structure set, selector of the reference protocols present in the data,
        and notes.
    """
    structure_set = diagnostics["structure_set"]
    references = diagnostics["references"]
    scored_label = references[diagnostics["scored_reference"]]["label"]

    return Div(
        [
            P(
                f"Structures: {structure_set['n_structures']:,} "
                f"{structure_set['label']}, the same for every reference protocol."
            ),
            Label("DFT reference protocol"),
            dcc.Dropdown(
                id=REFERENCE_DROPDOWN_ID,
                options=[
                    {"label": reference["label"], "value": key}
                    for key, reference in references.items()
                ],
                value=diagnostics["scored_reference"],
                clearable=False,
                style={"width": "300px", "marginBottom": "10px"},
            ),
            Div(
                "Additional DFT reference protocols on the same geometries are "
                "planned.",
                style=INSTRUCTION_STYLE,
            ),
            P(
                f"The main table uses {scored_label}. The selected protocol sets "
                "the parity plots and the diagnostics below the table."
            ),
        ],
        style={"marginBottom": "20px"},
    )


class MADBenchApp(BaseApp):
    """
    MAD-BENCH benchmark app.

    Parameters
    ----------
    diagnostics
        Contents of the diagnostics file written by the analysis.
    **kwargs
        Arguments passed to ``BaseApp``.
    """

    def __init__(self, diagnostics: dict, **kwargs) -> None:
        """
        Store the diagnostics before the layout is built.

        Parameters
        ----------
        diagnostics
            Contents of the diagnostics file written by the analysis.
        **kwargs
            Arguments passed to ``BaseApp``.
        """
        self.diagnostics = diagnostics
        super().__init__(**kwargs)

    def build_layout(self) -> Div:
        """
        Build the standard layout with the reference selector above the table.

        Returns
        -------
        Div
            Benchmark layout.
        """
        layout = super().build_layout()
        children = list(layout.children)

        # the selector goes after the documentation link, above the main table
        position = next(
            index
            for index, child in enumerate(children)
            if getattr(child, "className", None) == "mlpeg-doc-link"
        )
        children.insert(position + 1, build_reference_selector(self.diagnostics))
        layout.children = children

        return layout

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        references = self.diagnostics["references"]
        density_plots = {}

        # parity plots are precomputed per reference, so switching never reruns models
        for key in references:
            density_plots[key] = {}

            for model in MODELS:
                plots = {}

                for column, name in PLOTTED_METRICS.items():
                    plot = read_density_plot_for_model(
                        filename=DATA_PATH / f"figure_{name}_density_{key}.json",
                        model=model,
                        id=figure_id(key, model, name),
                    )

                    if plot is not None:
                        plots[column] = plot

                density_plots[key][model] = plots

        register_plot_download_callbacks()
        register_plot_settings_callbacks()

        # clicking a point shows the structure that component belongs to
        for key in references:
            point_sources = read_json(DATA_PATH / f"density_structures_{key}.json")

            for column, name in PLOTTED_METRICS.items():
                for model, sources in point_sources[name].items():
                    if column not in density_plots[key].get(model, {}):
                        continue

                    struct_from_scatter(
                        scatter_id=figure_id(key, model, name),
                        struct_id=STRUCT_ID,
                        structs=[
                            f"{ASSETS_PREFIX}/structures/{source}.xyz"
                            for source in sources
                        ],
                        mode="struct",
                    )

        @callback(
            Output(CELL_STORE_ID, "data"),
            Output(self.table_id, "active_cell"),
            Input(self.table_id, "active_cell"),
            prevent_initial_call=True,
        )
        def remember_cell(active_cell: dict | None) -> tuple[dict, None]:
            """
            Remember the clicked cell and clear it so it can be clicked again.

            Parameters
            ----------
            active_cell
                Clicked cell in the main table.

            Returns
            -------
            tuple[dict, None]
                Model and column of the clicked cell, and a cleared active cell.
            """
            if not active_cell:
                raise PreventUpdate

            return {
                "model": active_cell.get("row_id"),
                "column": active_cell.get("column_id"),
            }, None

        @callback(
            Output(PLOT_ID, "children"),
            Input(CELL_STORE_ID, "data"),
            Input(REFERENCE_DROPDOWN_ID, "value"),
        )
        def show_plot(cell: dict | None, reference_key: str) -> Div:
            """
            Show the parity plot of the clicked model for the selected reference.

            Parameters
            ----------
            cell
                Model and column of the last clicked cell.
            reference_key
                Selected reference protocol.

            Returns
            -------
            Div
                Parity plot, or a hint if no Force or Stress MAE cell has been clicked.
            """
            if not cell or cell["column"] not in PLOTTED_METRICS:
                return Div(TABLE_HINT, style=INSTRUCTION_STYLE)

            model_plots = density_plots[reference_key].get(cell["model"], {})
            plot = model_plots.get(cell["column"])

            if plot is None:
                return Div("No data available for this model.")

            return plot_with_download_controls(plot)

        @callback(
            Output(DIAGNOSTICS_TABLE_ID, "columns"),
            Output(DIAGNOSTICS_TABLE_ID, "data"),
            Output(DIAGNOSTICS_NOTE_ID, "children"),
            Input(REFERENCE_DROPDOWN_ID, "value"),
        )
        def show_diagnostics(reference_key: str) -> tuple[list, list, str]:
            """
            Show the unscored diagnostics for the selected reference.

            Parameters
            ----------
            reference_key
                Selected reference protocol.

            Returns
            -------
            tuple[list, list, str]
                Table columns, rows and note.
            """
            reference_diagnostics = references[reference_key]
            columns, rows = diagnostics_table_content(reference_diagnostics)

            return columns, rows, diagnostics_note(reference_diagnostics)


def get_app() -> MADBenchApp:
    """
    Get MAD-BENCH app instance with layout and callbacks.

    Returns
    -------
    MADBenchApp
        Benchmark layout and callback registration.
    """
    diagnostics = read_json(DATA_PATH / "mad_bench_diagnostics.json")
    scored_diagnostics = diagnostics["references"][diagnostics["scored_reference"]]
    columns, rows = diagnostics_table_content(scored_diagnostics)

    return MADBenchApp(
        diagnostics=diagnostics,
        name=BENCHMARK_NAME,
        description=(
            "Force and stress errors on 2,000 structures sampled from the MAD-1.6 "
            "test split, against its r2SCAN reference labels."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "mad_bench_metrics_table.json",
        extra_components=[
            dcc.Store(id=CELL_STORE_ID),
            Div(Div(TABLE_HINT, style=INSTRUCTION_STYLE), id=PLOT_ID),
            Div(id=STRUCT_ID),
            Div(
                [
                    P(diagnostics_note(scored_diagnostics), id=DIAGNOSTICS_NOTE_ID),
                    DataTable(
                        id=DIAGNOSTICS_TABLE_ID,
                        columns=columns,
                        data=rows,
                        style_table={"overflowX": "auto"},
                        style_header={"whiteSpace": "pre-line", "height": "auto"},
                        style_cell={"textAlign": "left", "padding": "4px 8px"},
                        editable=False,
                    ),
                ]
            ),
        ],
        info_path=DATA_PATH / "info.json",
    )
