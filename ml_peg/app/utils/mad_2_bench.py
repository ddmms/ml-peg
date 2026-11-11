"""Shared apps for the MAD-2 DFT label benchmarks."""

from __future__ import annotations

import json

from dash import Input, Output, callback, dcc, html
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from dash.dcc import Store
from dash.html import Div, Label

from ml_peg.analysis.utils.utils import get_table_style
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp

DATA_PATH = APP_ROOT / "data" / "mad_2_bench"

PROPERTIES = {
    "forces": ("Forces", "meV/Å", 1, "meV_per_A"),
    "energy": ("Energy", "meV/atom", 1, "meV_per_atom"),
    "stress": ("Stress", "meV/Å³", 2, "meV_per_A3"),
}

STATISTICS = {"mae": "MAE", "rmse": "RMSE", "p95": "p95", "max": "Max"}
SHOWN_STATISTICS = ("mae", "rmse", "p95")

CLASS_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "Bulk", "value": "bulk"},
    {"label": "Surfaces", "value": "surface"},
    {"label": "2D layers", "value": "2D"},
    {"label": "Molecules", "value": "molecule"},
    {"label": "Clusters", "value": "cluster"},
]

MODE_OPTIONS = [
    {"label": "All labelled frames", "value": "available"},
    {"label": "Common frames", "value": "common"},
]

MOLECULAR_DOMAIN = ("molecule", "cluster")

FRAMEWORK_ID = "mad-2-bench"

DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/mad-2-bench.html"

ENERGY_NOTE = (
    "Energies only where POTCAR, Hubbard U and energy definition match the model's "
    "training data."
)

SELECTOR_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": "12px"}
LINE_STYLE = {"fontSize": "14px", "margin": "10px 0"}
NOTE_STYLE = {"fontSize": "13px", "margin": "6px 0", "color": "#555555"}
CELL_STYLE = {"fontSize": "13px", "padding": "4px 10px", "textAlign": "left"}
HEADER_STYLE = {"fontWeight": "bold", "whiteSpace": "normal", "height": "auto"}
PINNED_STYLE = {
    "if": {"column_id": "MLIP"},
    "minWidth": "230px",
    "maxWidth": "300px",
    "position": "sticky",
    "left": 0,
    "zIndex": 1,
    "backgroundColor": "white",
}


def energy_tooltips(cells: dict) -> dict:
    """
    Get the energy tooltip of each reference cell.

    Parameters
    ----------
    cells
        Comparison cells of one model.

    Returns
    -------
    dict
        One tooltip per reference column.
    """
    tooltips = {}

    for reference, cell in cells["references"].items():
        excluded = ", ".join(
            f"{status} {count}"
            for status, count in sorted(cell.get("energy_excluded", {}).items())
        )
        text = f"{cell.get('energy_frames', 0)} compatible frames"
        tooltips[reference] = {
            "value": text + (f", excluded: {excluded}" if excluded else ""),
            "type": "markdown",
        }

    return tooltips


def dropdown(identifier: str, label: str, options: list, value: str) -> Div:
    """
    Build a labelled dropdown.

    Parameters
    ----------
    identifier
        Component ID.
    label
        Label above the dropdown.
    options
        Dropdown options.
    value
        Selected value.

    Returns
    -------
    Div
        Label and dropdown.
    """
    return Div(
        [
            Label(label),
            dcc.Dropdown(
                id=identifier,
                options=options,
                value=value,
                clearable=False,
                style={"width": "230px"},
            ),
        ]
    )


def subset_text(counts: dict[str, int]) -> str:
    """
    Get the MAD-2 subsets line.

    Parameters
    ----------
    counts
        Labelled frames per subset.

    Returns
    -------
    str
        One line, for example "MAD-2 subsets: binary_random (100), mc3d (97)".
    """
    subsets = ", ".join(
        f"{subset} ({count})" for subset, count in sorted(counts.items())
    )

    return f"MAD-2 subsets: {subsets}" if subsets else ""


def number(nully: str, precision: int) -> Format:
    """
    Get a fixed point format with a label for empty cells.

    Parameters
    ----------
    nully
        Text shown instead of an empty value.
    precision
        Digits after the decimal point.

    Returns
    -------
    Format
        Dash table number format.
    """
    return Format(precision=precision, scheme=Scheme.fixed, nully=nully)


def hidden_stores(table_id: str) -> list[Store]:
    """
    Get the empty stores the shared table callbacks expect.

    Parameters
    ----------
    table_id
        ID of the benchmark table.

    Returns
    -------
    list[Store]
        Empty stores.
    """
    return [
        Store(id=f"{table_id}-{suffix}", data=[] if "data" in suffix else {})
        for suffix in (
            "weight-store",
            "thresholds-store",
            "computed-store",
            "raw-data-store",
            "raw-tooltip-store",
        )
    ]


def domain_toggle(identifier: str) -> Div:
    """
    Build the out-of-domain toggle.

    Parameters
    ----------
    identifier
        Component ID.

    Returns
    -------
    Div
        Checklist with one option, off by default.
    """
    return Div(
        dcc.Checklist(
            id=identifier,
            options=[{"label": " Include out-of-domain models", "value": "include"}],
            value=[],
        ),
        style={"alignSelf": "flex-end", "marginBottom": "8px"},
    )


def in_domain(molecular: bool, class_name: str) -> bool:
    """
    Check whether a structure class is in the domain of a model.

    Parameters
    ----------
    molecular
        Whether the model was trained only on isolated molecules.
    class_name
        Selected structure class.

    Returns
    -------
    bool
        False for molecule-only models on periodic structures.
    """
    return not molecular or class_name in MOLECULAR_DOMAIN


def model_tooltip(model: dict, extra: list[str]) -> dict:
    """
    Get the tooltip of a model name.

    Parameters
    ----------
    model
        Model entry from models.json.
    extra
        Further lines, for example the energy comparison.

    Returns
    -------
    dict
        Markdown tooltip for the model cell.
    """
    lines = [
        f"**{model['name']}**",
        f"Identifier: `{model['identifier']}`",
        f"Checkpoint: {model['checkpoint']}",
        f"Training: {model['training_data']}",
        *extra,
    ]

    return {"value": "  \n".join(lines), "type": "markdown"}


def data_table(identifier: str, **kwargs) -> DataTable:
    """
    Build a table with the model column pinned.

    Parameters
    ----------
    identifier
        Component ID.
    **kwargs
        Further DataTable arguments.

    Returns
    -------
    DataTable
        Dash table.
    """
    return DataTable(
        id=identifier,
        style_table={"minWidth": "100%", "overflowX": "auto"},
        style_cell=CELL_STYLE,
        style_cell_conditional=[PINNED_STYLE],
        style_header=HEADER_STYLE,
        style_header_conditional=[{**PINNED_STYLE, "zIndex": 2}],
        **kwargs,
    )


def rank_colours(rows: list[dict], columns: list[str]) -> list[dict]:
    """
    Colour each column by the rank of the models in the current view.

    Parameters
    ----------
    rows
        Table rows with an ``MLIP`` field.
    columns
        Numeric columns to colour, lower is better.

    Returns
    -------
    list[dict]
        Conditional styles, empty cells keep their text label.
    """
    if not rows:
        return []

    scored = [dict(row) for row in rows]

    for column in columns:
        values = sorted({row[column] for row in rows if row[column] is not None})

        for row in scored:
            if row[column] is None:
                continue

            rank = values.index(row[column])
            row[column] = rank / (len(values) - 1) if len(values) > 1 else 0.0

    style = []

    for column in columns:
        style += get_table_style(
            rows, scored_data=scored, normalized=True, all_cols=False, col_names=column
        )

    return [entry for entry in style if entry.get("color") != "transparent"]


class Mad2ErrorsApp(BaseApp):
    """
    Errors of every model against one reference and one property at a time.

    Parameters
    ----------
    name
        Benchmark name shown in the app.
    description
        Benchmark description shown in the app.
    """

    def __init__(self, name: str, description: str) -> None:
        """
        Initialise the app.

        Parameters
        ----------
        name
            Benchmark name shown in the app.
        description
            Benchmark description shown in the app.
        """
        data_path = DATA_PATH / "errors"
        selections = json.loads((data_path / "selections.json").read_text())
        self.rows = selections["rows"]
        self.summaries = selections["summaries"]
        self.references = json.loads((data_path / "references.json").read_text())
        self.models = json.loads((data_path / "models.json").read_text())
        self.subsets = json.loads((data_path / "subsets.json").read_text())

        super().__init__(
            name=name,
            description=description,
            table_path=data_path / "errors_metrics_table.json",
            extra_components=[],
            docs_url=DOCS_URL,
            framework_ids=FRAMEWORK_ID,
            include_ml_peg=False,
        )

    def build_layout(self) -> Div:
        """
        Build the selectors and the table.

        Returns
        -------
        Div
            Benchmark layout.
        """
        references = [{"label": n, "value": k} for k, n in self.references.items()]
        properties = [
            {"label": value[0], "value": key} for key, value in PROPERTIES.items()
        ]

        selectors = Div(
            [
                dropdown(
                    f"{self.name}-reference", "Reference", references, "mad/r2scan"
                ),
                dropdown(f"{self.name}-property", "Property", properties, "forces"),
                dropdown(f"{self.name}-class", "Structures", CLASS_OPTIONS, "bulk"),
                dropdown(f"{self.name}-mode", "Frames", MODE_OPTIONS, "available"),
                domain_toggle(f"{self.name}-domain"),
            ],
            style=SELECTOR_STYLE,
        )

        table = data_table(
            f"{self.name}-mad-table",
            columns=[],
            data=[],
            sort_action="native",
            sort_by=[{"column_id": "mae", "direction": "asc"}],
            tooltip_delay=0,
            tooltip_duration=None,
            page_size=120,
        )

        return Div(
            [
                html.H2(self.name),
                html.P(self.description),
                html.A(
                    "View documentation \u2192",
                    href=self.docs_url,
                    target="_blank",
                    className="mlpeg-doc-link",
                ),
                selectors,
                Div(id=f"{self.name}-subsets", style=NOTE_STYLE),
                Div(id=f"{self.name}-summary", style=LINE_STYLE),
                table,
                *hidden_stores(self.table_id),
            ]
        )

    def register_callbacks(self) -> None:
        """Register the selector callbacks."""

        @callback(
            Output(f"{self.name}-subsets", "children"),
            Input(f"{self.name}-reference", "value"),
            Input(f"{self.name}-class", "value"),
        )
        def show_subsets(reference: str, class_name: str) -> str:
            """
            List the MAD-2 subsets of the selected reference and structures.

            Parameters
            ----------
            reference
                Selected reference key.
            class_name
                Selected structure class.

            Returns
            -------
            str
                The subsets with their frame counts.
            """
            return subset_text(self.subsets.get(reference, {}).get(class_name, {}))

        @callback(
            Output(f"{self.name}-mad-table", "data"),
            Output(f"{self.name}-mad-table", "columns"),
            Output(f"{self.name}-mad-table", "tooltip_data"),
            Output(f"{self.name}-mad-table", "style_data_conditional"),
            Output(f"{self.name}-summary", "children"),
            Input(f"{self.name}-reference", "value"),
            Input(f"{self.name}-property", "value"),
            Input(f"{self.name}-class", "value"),
            Input(f"{self.name}-mode", "value"),
            Input(f"{self.name}-domain", "value"),
            Input("selected-models-store", "data"),
        )
        def update_table(
            reference: str,
            prop: str,
            class_name: str,
            mode: str,
            domain: list[str],
            selected_models: list[str] | None,
        ) -> tuple[list[dict], list[dict], list[dict], list[dict], list]:
            """
            Show one property of every model for the selected reference and frames.

            Parameters
            ----------
            reference
                Selected reference key.
            prop
                Selected property, forces, energy or stress.
            class_name
                Selected structure class.
            mode
                All labelled frames or common frames.
            domain
                Whether out-of-domain models are included.
            selected_models
                Models selected in the global model filter, None before it loads.

            Returns
            -------
            tuple[list[dict], list[dict], list[dict], list[dict], list]
                Rows, columns, model tooltips, colours and the summary line.
            """
            key = f"{reference}|{class_name}|all|all"

            if key not in self.summaries:
                return [], [], [], [], "No labels in this selection."

            summary = self.summaries[key]
            _, unit, precision, _ = PROPERTIES[prop]
            rows = self.rows.get(f"{key}|{mode}", [])

            if prop == "stress" and not summary["stress_labels"]:
                return (
                    [],
                    [],
                    [],
                    [],
                    "No stress labels here, stress is a label for bulk only.",
                )

            if prop == "energy":
                rows = [row for row in rows if row["comparable"]]

            if "include" not in domain:
                rows = [
                    row
                    for row in rows
                    if in_domain(self.models[row["model"]]["molecular"], class_name)
                ]

            if selected_models is not None:
                rows = [row for row in rows if row["model"] in selected_models]

            summary_children = []

            if mode == "common":
                common = summary["common"][prop]
                compared = sum(1 for row in rows if row[f"{prop}_mae"] is not None)
                summary_children.append(
                    Div(
                        f"{compared} models compared on the same "
                        f"{common['frames']} structures.",
                        style=LINE_STYLE,
                    )
                )

            if prop == "energy":
                summary_children.append(Div(ENERGY_NOTE, style=NOTE_STYLE))

            empty = (
                "-"
                if mode == "common"
                else ("no stress" if prop == "stress" else "no prediction")
            )
            columns = [
                {"id": "MLIP", "name": "Model"},
                {"id": "coverage", "name": "Coverage"},
            ] + [
                {
                    "id": stat,
                    "name": f"{STATISTICS[stat]} [{unit}]",
                    "type": "numeric",
                    "format": number(empty, precision),
                }
                for stat in SHOWN_STATISTICS
            ]

            table = [
                {
                    "id": row["model"],
                    "MLIP": self.models[row["model"]]["name"],
                    "coverage": row[f"{prop}_coverage"],
                    **{stat: row[f"{prop}_{stat}"] for stat in SHOWN_STATISTICS},
                }
                for row in rows
            ]
            tooltips = [
                {
                    "MLIP": model_tooltip(
                        self.models[row["model"]], [f"Energies: {row['energies']}."]
                    )
                }
                for row in rows
            ]
            colours = rank_colours(table, list(SHOWN_STATISTICS))

            return table, columns, tooltips, colours, summary_children


class Mad2ComparisonApp(BaseApp):
    """
    One property of every model against the six references on matched frames.

    Parameters
    ----------
    name
        Benchmark name shown in the app.
    description
        Benchmark description shown in the app.
    """

    def __init__(self, name: str, description: str) -> None:
        """
        Initialise the app.

        Parameters
        ----------
        name
            Benchmark name shown in the app.
        description
            Benchmark description shown in the app.
        """
        data_path = DATA_PATH / "comparison"
        comparison = json.loads((data_path / "comparison.json").read_text())
        self.entries = comparison["entries"]
        self.references = json.loads((data_path / "references.json").read_text())
        self.models = json.loads((data_path / "models.json").read_text())
        self.disagreement = json.loads((data_path / "disagreement.json").read_text())

        super().__init__(
            name=name,
            description=description,
            table_path=data_path / "comparison_metrics_table.json",
            extra_components=[],
            docs_url=DOCS_URL,
            framework_ids=FRAMEWORK_ID,
            include_ml_peg=False,
        )

    def disagreement_table(self) -> Div:
        """
        Build the force disagreement between the references on bulk frames.

        Returns
        -------
        Div
            Heading and matrix.
        """
        matrix = self.disagreement["forces_mae_meV_per_A"]
        rows = [
            {
                "MLIP": name,
                **{
                    other: matrix[key].get(other) if other != key else None
                    for other in self.references
                },
            }
            for key, name in self.references.items()
        ]
        columns = [{"id": "MLIP", "name": "Force MAE [meV/Å]"}] + [
            {
                "id": key,
                "name": name,
                "type": "numeric",
                "format": number("-", 1),
            }
            for key, name in self.references.items()
        ]
        return Div(
            [
                html.H3(
                    "Force difference between references, "
                    f"{self.disagreement['shared_frames']} bulk frames"
                ),
                data_table(
                    f"{self.name}-disagreement-table",
                    columns=columns,
                    data=rows,
                ),
            ],
            style={"marginTop": "24px"},
        )

    def build_layout(self) -> Div:
        """
        Build the selectors, the comparison table and the disagreement matrix.

        Returns
        -------
        Div
            Benchmark layout.
        """
        properties = [
            {"label": value[0], "value": key} for key, value in PROPERTIES.items()
        ]
        statistics = [
            {"label": label, "value": key} for key, label in STATISTICS.items()
        ]

        selectors = Div(
            [
                dropdown(f"{self.name}-property", "Property", properties, "forces"),
                dropdown(f"{self.name}-statistic", "Metric", statistics, "mae"),
                dropdown(f"{self.name}-class", "Structures", CLASS_OPTIONS, "bulk"),
                domain_toggle(f"{self.name}-domain"),
            ],
            style=SELECTOR_STYLE,
        )

        table = data_table(
            f"{self.name}-comparison-table",
            columns=[],
            data=[],
            sort_action="native",
            filter_action="native",
            filter_options={"case": "insensitive"},
            tooltip_delay=0,
            tooltip_duration=None,
            page_size=120,
        )

        return Div(
            [
                html.H2(self.name),
                html.P(self.description),
                html.A(
                    "View documentation \u2192",
                    href=self.docs_url,
                    target="_blank",
                    className="mlpeg-doc-link",
                ),
                selectors,
                Div(id=f"{self.name}-summary", style=LINE_STYLE),
                table,
                self.disagreement_table(),
                *hidden_stores(self.table_id),
            ]
        )

    def register_callbacks(self) -> None:
        """Register the selector callbacks."""

        @callback(
            Output(f"{self.name}-comparison-table", "data"),
            Output(f"{self.name}-comparison-table", "columns"),
            Output(f"{self.name}-comparison-table", "tooltip_data"),
            Output(f"{self.name}-comparison-table", "style_data_conditional"),
            Output(f"{self.name}-summary", "children"),
            Input(f"{self.name}-property", "value"),
            Input(f"{self.name}-statistic", "value"),
            Input(f"{self.name}-class", "value"),
            Input(f"{self.name}-domain", "value"),
            Input("selected-models-store", "data"),
        )
        def update_table(
            prop: str,
            statistic: str,
            class_name: str,
            domain: list[str],
            selected_models: list[str] | None,
        ) -> tuple[list[dict], list[dict], list[dict], list[dict], list]:
            """
            Show the selected property of every model for the six references.

            Parameters
            ----------
            prop
                Selected property, forces, energy or stress.
            statistic
                Selected statistic, MAE, RMSE, p95 or max.
            class_name
                Selected structure class.
            domain
                Whether out-of-domain models are included.
            selected_models
                Models selected in the global model filter, None before it loads.

            Returns
            -------
            tuple[list[dict], list[dict], list[dict], list[dict], list]
                Rows, columns, tooltips, colours and the summary line.
            """
            entry = self.entries.get(f"{class_name}|all|full")

            if entry is None:
                return (
                    [],
                    [],
                    [],
                    [],
                    "No frame is labelled by all six references here.",
                )

            _, unit, precision, suffix = PROPERTIES[prop]
            field = f"{prop}_{statistic}_{suffix}"
            ranked = []
            others = []

            for model, cells in sorted(
                entry["models"].items(), key=lambda item: self.models[item[0]]["name"]
            ):
                if prop not in cells["properties"]:
                    continue

                if selected_models is not None and model not in selected_models:
                    continue

                if "include" not in domain and not in_domain(
                    self.models[model]["molecular"], class_name
                ):
                    continue

                used = cells["properties"][prop]
                name = self.models[model]["name"]
                row = {
                    "id": model,
                    "MLIP": name
                    if used["in_cohort"]
                    else f"{name} ({used['frames']} frames)",
                    **{
                        key: cells["references"][key].get(field)
                        for key in self.references
                    },
                }
                (ranked if used["in_cohort"] else others).append(row)

            rows = ranked + others

            summary = []

            if ranked and prop != "energy":
                shared = entry["models"][ranked[0]["id"]]["properties"][prop]["frames"]
                summary.append(
                    Div(
                        f"{len(ranked)} models compared on the same {shared} "
                        "structures.",
                        style=LINE_STYLE,
                    )
                )

            if prop == "energy":
                summary.append(Div(ENERGY_NOTE, style=NOTE_STYLE))

            empty = "-" if prop == "energy" else "no prediction"
            columns = [
                {"id": "MLIP", "name": f"Model, {STATISTICS[statistic]} [{unit}]"}
            ]

            columns += [
                {
                    "id": key,
                    "name": name,
                    "type": "numeric",
                    "format": number(empty, precision),
                }
                for key, name in self.references.items()
            ]

            tooltips = [
                {
                    "MLIP": model_tooltip(self.models[row["id"]], []),
                    **(
                        energy_tooltips(entry["models"][row["id"]])
                        if prop == "energy"
                        else {}
                    ),
                }
                for row in rows
            ]
            colours = rank_colours(ranked, list(self.references))

            return rows, columns, tooltips, colours, summary
