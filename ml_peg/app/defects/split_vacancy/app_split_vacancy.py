"""Run split vacancy benchmark app."""

from __future__ import annotations

from functools import lru_cache
from operator import itemgetter

from ase.io import read
from dash import Input, Output, callback
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import B, Div, Iframe, P, Span

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    _register_point_highlight,
    plot_from_table_column,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Split vacancy"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/defects.html#split-vacancy"
)
DATA_PATH = APP_ROOT / "data" / "defects" / "split_vacancy"
INFO_PATH = DATA_PATH / "info.json"

# for dash, assets/ is equivalent to APP_ROOT/data/
STRUCTS_URL = "/assets/defects/split_vacancy"

# Normalised max-distance below which an MLIP-relaxed structure counts as matching
# the DFT reference. Keep in sync with STOL in
# ml_peg/analysis/defects/split_vacancy/analyse_split_vacancy.py.
STOL = 0.25

XYZ_NAMES = {"NV": "normal_vacancy.xyz", "SV": "split_vacancy.xyz"}
VACANCY_LABELS = {"NV": "normal vacancy", "SV": "split vacancy"}

IFRAME_STYLE = {
    "height": "550px",
    "width": "100%",
    "border": "1px solid #ddd",
    "borderRadius": "5px",
}
GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
    "gap": "8px",
}
CAPTION_STYLE = {
    "fontSize": "1.15rem",
    "borderLeft": "4px solid #636efa",
    "paddingLeft": "10px",
    "margin": "8px 0 14px 0",
}
NOTE_STYLE = {"color": "#666", "fontSize": "0.9rem", "margin": "0 0 10px 10px"}
SUBTITLE_STYLE = {"color": "#666", "fontSize": "0.9rem"}


@lru_cache(maxsize=512)
def _read_frames(
    functional: str, source_dir: str, material_dir: str, cation: str, vac_type: str
) -> tuple[dict, ...]:
    """
    Read the energy and structure match of every candidate structure in one file.

    Parameters
    ----------
    functional
        DFT functional (``"pbe"`` or ``"pbesol"``), used to determine structure paths.
    source_dir
        Directory holding the structures, ``"ref"`` for the DFT reference, otherwise
        the name of the MLIP that relaxed them.
    material_dir
        Directory name of the host compound, ``"{formula}-{mp_id}"``.
    cation
        Symbol of the vacant cation.
    vac_type
        Vacancy type, ``"NV"`` or ``"SV"``.

    Returns
    -------
    tuple[dict, ...]
        One dictionary per frame, with keys ``"index"`` (frame), ``"n_frames"``
        (frames in the file), ``"energy"`` (eV), ``"matched"`` (whether the frame
        matches the DFT geometry) and ``"max_dist"`` (normalised max distance to the
        reference, or `None` for the reference itself). Empty if the file is missing.
    """
    path = (
        DATA_PATH
        / functional
        / source_dir
        / material_dir
        / cation
        / XYZ_NAMES[vac_type]
    )
    if not path.exists():
        return ()

    structs = read(path, ":")
    if source_dir == "ref":
        energies = [float(struct.info["ref_energy"]) for struct in structs]
        max_dists = [None] * len(structs)
        matches = [True] * len(structs)
    else:
        energies = [float(struct.info["relaxed_energy"]) for struct in structs]
        max_dists = [float(struct.info["ref_max_distance"]) for struct in structs]
        matches = [max_dist < STOL for max_dist in max_dists]

    return tuple(
        {
            "index": index,
            "n_frames": len(structs),
            "energy": energies[index],
            "matched": matches[index],
            "max_dist": max_dists[index],
        }
        for index in range(len(structs))
    )


@lru_cache(maxsize=256)
def _select_frames(
    functional: str, model_name: str, material_dir: str, cation: str
) -> dict[tuple[str, str], dict]:
    """
    Find the structures whose energies set the formation energy of one scatter point.

    The formation energy is ``min(E_SV) - min(E_NV)``, with the minima taken over the
    structures that relaxed to the DFT geometry, so each point summarises four
    structures: the reference and MLIP minima of both vacancy types.

    Parameters
    ----------
    functional
        DFT functional (``"pbe"`` or ``"pbesol"``), used to determine structure paths.
    model_name
        Name of the MLIP whose relaxed structures are compared to the reference.
    material_dir
        Directory name of the host compound, ``"{formula}-{mp_id}"``.
    cation
        Symbol of the vacant cation.

    Returns
    -------
    dict[tuple[str, str], dict]
        Map of ``(source, vacancy type)``, where source is ``"ref"`` or ``"mlip"`` and
        vacancy type is ``"NV"`` or ``"SV"``, to the frame chosen for that pair, as
        returned by `_read_frames`. Pairs with no structure file are omitted.
    """
    frames = {}

    for source, source_dir in (("ref", "ref"), ("mlip", model_name)):
        for vac_type in XYZ_NAMES:
            candidates = _read_frames(
                functional, source_dir, material_dir, cation, vac_type
            )
            if not candidates:
                continue

            # Only matched structures contribute to the metric, but fall back to all
            # frames so an unmatched pair can still be inspected.
            matched = [frame for frame in candidates if frame["matched"]]
            frames[source, vac_type] = min(
                matched or candidates, key=itemgetter("energy")
            )

    return frames


def _struct_panel(struct_url: str, title: str, vac_type: str, frame: dict) -> Div:
    """
    Build a labelled WEAS viewer for one structure of a material-cation pair.

    Parameters
    ----------
    struct_url
        URL of the structure file to visualise.
    title
        Bold label describing where the structure came from.
    vac_type
        Vacancy type, ``"NV"`` or ``"SV"``.
    frame
        Frame information for the structure, as returned by `_read_frames`.

    Returns
    -------
    Div
        Title, subtitle and WEAS viewer opened at the relevant frame.
    """
    subtitle = (
        f"{VACANCY_LABELS[vac_type]} · frame {frame['index'] + 1}/{frame['n_frames']}"
        f" · E = {frame['energy']:.3f} eV"
    )
    if frame["max_dist"] is not None:
        subtitle += f" · max dist = {frame['max_dist']:.2f}"
    if not frame["matched"]:
        subtitle += " · no match"

    return Div(
        [
            P([B(title), Span(f" — {subtitle}", style=SUBTITLE_STYLE)]),
            Iframe(
                srcDoc=generate_weas_html(
                    struct_url, mode="traj", index=frame["index"]
                ),
                style=IFRAME_STYLE,
            ),
        ]
    )


def _struct_panels(
    functional: str,
    model_name: str,
    material_dir: str,
    cation: str,
    frames: dict[tuple[str, str], dict],
) -> list[Div]:
    """
    Build the reference and MLIP viewers for each selected frame.

    Parameters
    ----------
    functional
        DFT functional (``"pbe"`` or ``"pbesol"``), used to determine structure paths.
    model_name
        Name of the MLIP whose relaxed structures are compared to the reference.
    material_dir
        Directory name of the host compound, ``"{formula}-{mp_id}"``.
    cation
        Symbol of the vacant cation.
    frames
        Frame to show for each ``(source, vacancy type)``, as returned by
        `_select_frames`.

    Returns
    -------
    list[Div]
        Viewers ordered for a two-column grid, with the reference in the left column
        and the MLIP in the right, and normal vacancies above split vacancies.
    """
    titles = {"ref": "DFT relaxed (reference)", "mlip": f"MLIP relaxed ({model_name})"}
    dirs = {"ref": "ref", "mlip": model_name}
    prefix = f"{STRUCTS_URL}/{functional}"

    return [
        _struct_panel(
            f"{prefix}/{dirs[source]}/{material_dir}/{cation}/{XYZ_NAMES[vac_type]}",
            titles[source],
            vac_type,
            frames[source, vac_type],
        )
        for vac_type in XYZ_NAMES
        for source in ("ref", "mlip")
        if (source, vac_type) in frames
    ]


def struct_pair_from_violin(
    violin_id: str,
    struct_id: str,
    functional: str,
) -> None:
    """
    Register callback to show ref and MLIP structures when a violin point is clicked.

    Each point of the max dist violin is one candidate structure, so clicking one
    shows the DFT-relaxed reference beside the MLIP-relaxed structure it is compared
    against. Both viewers open at the clicked structure, but keep the full trajectory
    so the other candidates of that vacancy type can be stepped through.

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
            Visualised structures on plot click.
        """
        if not click_data:
            return Div("Click on a point to view structures.")

        point = click_data["points"][0]
        model_name = point["x"]
        mp_id, formula, cation, vac_type = point["customdata"][:4]
        frame_id = int(point["customdata"][4])
        material_dir = f"{formula}-{mp_id}"

        frames = {}
        for source, source_dir in (("ref", "ref"), ("mlip", model_name)):
            candidates = _read_frames(
                functional, source_dir, material_dir, cation, vac_type
            )
            if frame_id >= len(candidates):
                return Div(f"Structures unavailable for {formula} ({mp_id}), {cation}.")
            frames[source, vac_type] = candidates[frame_id]

        max_dist = frames["mlip", vac_type]["max_dist"]
        return Div(
            [
                P(
                    B(
                        f"Showing {formula} ({mp_id}) — {cation} vacancy — "
                        f"{VACANCY_LABELS[vac_type]} — max dist {max_dist:.3f} "
                        f"vs match threshold {STOL}"
                    ),
                    style=CAPTION_STYLE,
                ),
                P(
                    "Showing the clicked structure, relaxed from the same starting "
                    "geometry by DFT and by the MLIP. Step through the trajectory to "
                    "see the other candidate structures of this type.",
                    style=NOTE_STYLE,
                ),
                Div(
                    _struct_panels(
                        functional, model_name, material_dir, cation, frames
                    ),
                    style=GRID_STYLE,
                ),
            ]
        )


def struct_quad_from_scatter(
    scatter_id: str,
    struct_id: str,
    functional: str,
    curve_models: list[str | None],
) -> None:
    """
    Register callback to show the four structures behind a formation energy point.

    Each point of the formation energy parity plot compares ``min(E_SV) - min(E_NV)``
    for the MLIP and the DFT reference, so clicking one shows the reference and MLIP
    minima of both vacancy types, arranged as reference (left) against MLIP (right),
    and normal vacancy (top) against split vacancy (bottom). Each viewer opens at the
    frame that set the energy, but keeps the full trajectory so the other candidate
    structures for the pair can be stepped through.

    Parameters
    ----------
    scatter_id
        ID for Dash scatter plot being clicked.
    struct_id
        ID for Dash placeholder Div where structures will be visualised.
    functional
        DFT functional (``"pbe"`` or ``"pbesol"``), used to determine structure paths.
    curve_models
        Model plotted by each trace of the figure, in trace order. Traces that do not
        correspond to a model, such as the parity line, are `None`.
    """
    # Frames of the viewers are candidate structures, not scatter points, so stepping
    # through them must not move the highlight.
    _register_point_highlight(scatter_id, follow_frames=False)

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_structs(click_data):
        """
        Register callback to show structures when point clicked. See build_callbacks.py.

        Parameters
        ----------
        click_data
            Clicked data point in scatter plot.

        Returns
        -------
        Div
            Visualised structures on plot click.
        """
        if not click_data:
            return Div("Click on a point to view structures.")

        point = click_data["points"][0]
        curve_number = point["curveNumber"]
        model_name = (
            curve_models[curve_number] if curve_number < len(curve_models) else None
        )
        custom_data = point.get("customdata")
        if model_name is None or not custom_data:
            # Parity line, or the ring marking the previously clicked point.
            raise PreventUpdate

        mp_id, formula, cation = custom_data[0], custom_data[1], custom_data[2]
        material_dir = f"{formula}-{mp_id}"

        frames = _select_frames(functional, model_name, material_dir, cation)
        if len(frames) < len(XYZ_NAMES) * 2:
            return Div(f"Structures unavailable for {formula} ({mp_id}), {cation}.")

        error = point["x"] - point["y"]
        return Div(
            [
                P(
                    B(
                        f"Showing {formula} ({mp_id}) — {cation} vacancy — formation "
                        f"energy {point['x']:.3f} eV vs DFT {point['y']:.3f} eV "
                        f"({error:+.3f} eV)"
                    ),
                    style=CAPTION_STYLE,
                ),
                P(
                    "Showing the lowest-energy matched structures, which set the "
                    "formation energy. Step through the trajectory to see the other "
                    "candidate structures for this material-cation pair.",
                    style=NOTE_STYLE,
                ),
                Div(
                    _struct_panels(
                        functional, model_name, material_dir, cation, frames
                    ),
                    style=GRID_STYLE,
                ),
            ]
        )


def _trace_models(graph: Graph) -> list[str | None]:
    """
    Get the model plotted by each trace of a parity figure, in trace order.

    Parameters
    ----------
    graph
        Dash graph of the parity plot, as loaded by `read_plot`.

    Returns
    -------
    list[str | None]
        Model name of each trace, or `None` for traces without one, such as the
        parity line.
    """
    if graph.figure is None:
        return []
    return [trace.name for trace in graph.figure.data]


class SplitVacancyApp(BaseApp):
    """Split vacancy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_pbesol = read_plot(
            DATA_PATH / "figure_formation_energies_pbesol.json",
            id=f"{BENCHMARK_NAME}-figure-scatter-pbesol",
        )
        scatter_pbe = read_plot(
            DATA_PATH / "figure_formation_energies_pbe.json",
            id=f"{BENCHMARK_NAME}-figure-scatter-pbe",
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
                "MAE (Oxides)": scatter_pbesol,
                "Spearman's (Oxides)": scatter_pbesol,
                "Match Rate (Oxides)": max_dist_violin_pbesol,
                "Max Dist (Oxides)": max_dist_violin_pbesol,
                "MAE (Nitrides)": scatter_pbe,
                "Spearman's (Nitrides)": scatter_pbe,
                "Match Rate (Nitrides)": max_dist_violin_pbe,
                "Max Dist (Nitrides)": max_dist_violin_pbe,
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

        struct_quad_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure-scatter-pbesol",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            functional="pbesol",
            curve_models=_trace_models(scatter_pbesol),
        )
        struct_quad_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure-scatter-pbe",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            functional="pbe",
            curve_models=_trace_models(scatter_pbe),
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
            "vacancies from fully ionised vacancies in nitrides (PBE) "
            "and oxides (PBEsol)."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "split_vacancy_metrics_table.json",
        info_path=INFO_PATH,
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )
