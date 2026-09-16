"""Test updating saved plots."""

from __future__ import annotations

import json

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pytest

from ml_peg import models
from ml_peg.analysis.utils.decorators import (
    cell_to_scatter,
    get_model_colour,
    merge_saved_models,
    merge_saved_traces,
    plot_density_scatter,
    plot_hist,
    plot_parity,
    plot_scatter,
    plot_violin,
)

pytestmark = pytest.mark.usefixtures("fake_models")


def get_trace_names(filename):
    """
    Get names of all named traces in a saved figure.

    Parameters
    ----------
    filename
        Filename of the saved figure.

    Returns
    -------
    list[str]
        Names of the saved traces, excluding unnamed traces such as parity lines.
    """
    return [trace.name for trace in pio.read_json(filename).data if trace.name]


def parity(filename, results):
    """
    Save a parity plot for pre-computed results.

    Parameters
    ----------
    filename
        Filename to save plot.
    results
        Reference and predicted values for each model.
    """
    plot_parity(filename=str(filename))(lambda: results)()


PLOTS = {
    "parity": (
        parity,
        {"ref": [1.0, 2.0], "model_1": [1.1, 2.1], "model_2": [1.2, 2.2]},
        {"ref": [1.0, 2.0], "model_2": [1.5, 2.5]},
    ),
    "hist": (
        lambda filename, results: plot_hist(bins=2, filename=filename)(
            lambda: results
        )(),
        {"model_1": [1.0, 2.0], "model_2": [2.0, 3.0]},
        {"model_2": [3.0, 4.0]},
    ),
    "scatter": (
        lambda filename, results: plot_scatter(filename=str(filename))(
            lambda: results
        )(),
        {"model_1": ([1.0, 2.0], [1.0, 2.0]), "model_2": ([1.0, 2.0], [2.0, 3.0])},
        {"model_2": ([1.0, 2.0], [3.0, 4.0])},
    ),
    "violin": (
        lambda filename, results: plot_violin(filename=str(filename))(
            lambda: results
        )(),
        {"model_1": [1.0, 2.0, 3.0], "model_2": [2.0, 3.0, 4.0]},
        {"model_2": [3.0, 4.0, 5.0]},
    ),
    "density": (
        lambda filename, results: plot_density_scatter(filename=str(filename))(
            lambda: results
        )(),
        {
            "model_1": {"ref": [1.0, 2.0], "pred": [1.1, 2.1]},
            "model_2": {"ref": [1.0, 2.0], "pred": [1.2, 2.2]},
        },
        {"model_2": {"ref": [1.0, 2.0], "pred": [1.5, 2.5]}},
    ),
}


@pytest.mark.parametrize("plot", PLOTS)
def test_update_plot(tmp_path, update_model_2, plot):
    """
    Test updating a plot preserves traces for models not being analysed.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved plot.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    plot
        Key of the plot being tested.
    """
    save_plot, all_results, model_2_results = PLOTS[plot]
    filename = tmp_path / f"{plot}.json"

    save_plot(filename, all_results)
    assert get_trace_names(filename) == ["model_1", "model_2"]

    save_plot(filename, model_2_results)
    assert get_trace_names(filename) == ["model_1", "model_2"]


@pytest.mark.parametrize("plot", PLOTS)
def test_overwrite_plot(tmp_path, monkeypatch, plot):
    """
    Test rerunning a subset of models without `--update` clears other models.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved plot.
    monkeypatch
        Pytest monkeypatch fixture.
    plot
        Key of the plot being tested.
    """
    save_plot, all_results, model_2_results = PLOTS[plot]
    filename = tmp_path / f"{plot}.json"

    save_plot(filename, all_results)

    monkeypatch.setattr(models, "current_models", "model_2")
    save_plot(filename, model_2_results)

    assert get_trace_names(filename) == ["model_2"]


def test_update_density_annotations(tmp_path, update_model_2):
    """
    Test annotations are preserved alongside density scatter traces.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved plot.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    save_plot, all_results, model_2_results = PLOTS["density"]
    filename = tmp_path / "density.json"

    save_plot(filename, all_results)
    save_plot(filename, model_2_results)

    meta = pio.read_json(filename).layout.meta
    assert meta["models"] == ["model_1", "model_2"]
    assert [annotation["text"] for annotation in meta["annotations"]] == [
        "model_1",
        "model_2",
    ]


def cell_scatter_bundle(models_data):
    """
    Build a data bundle for `cell_to_scatter`.

    Parameters
    ----------
    models_data
        Predicted value for each model.

    Returns
    -------
    dict
        Data bundle in the format expected by `cell_to_scatter`.
    """
    return {
        "metrics": {"mae": "MAE"},
        "models": {
            model: {"metrics": {"mae": {"points": [{"ref": 1.0, "pred": pred}]}}}
            for model, pred in models_data.items()
        },
    }


def test_update_cell_to_scatter(tmp_path, update_model_2):
    """
    Test updating cell scatter data preserves models not being analysed.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved data.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    filename = tmp_path / "cell_scatter.json"

    def save_data(models_data):
        """
        Save cell scatter data for pre-computed results.

        Parameters
        ----------
        models_data
            Predicted value for each model.
        """
        bundle = cell_scatter_bundle(models_data)
        cell_to_scatter(filename=filename)(lambda: bundle)()

    save_data({"model_1": 1.1, "model_2": 1.2})
    save_data({"model_2": 1.5})

    with open(filename) as fp:
        saved = json.load(fp)

    assert list(saved["models"]) == ["model_1", "model_2"]
    assert saved["models"]["model_1"]["metrics"]["mae"]["points"][0]["pred"] == 1.1
    assert saved["models"]["model_2"]["metrics"]["mae"]["points"][0]["pred"] == 1.5


def test_update_model_data(tmp_path, update_model_2):
    """
    Test updating data keyed by model preserves models not being analysed.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved data.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    filename = tmp_path / "figures.json"
    with open(filename, "w") as fp:
        json.dump({"model_1": {"value": 1.1}, "model_2": {"value": 1.2}}, fp)

    merged = merge_saved_models({"model_2": {"value": 1.5}}, filename)

    assert list(merged) == ["model_1", "model_2"]
    assert merged["model_1"] == {"value": 1.1}
    assert merged["model_2"] == {"value": 1.5}


def test_update_subplot_traces(tmp_path, update_model_2):
    """
    Test preserved traces remain assigned to their original subplot axes.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved plot.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    filename = tmp_path / "subplots.json"
    saved_fig = make_subplots(rows=1, cols=2)
    for col in (1, 2):
        saved_fig.add_trace(
            go.Scatter(x=[1.0], y=[1.0], name="model_1"), row=1, col=col
        )
    saved_fig.write_json(filename)

    fig = make_subplots(rows=1, cols=2)
    for col in (1, 2):
        fig.add_trace(go.Scatter(x=[1.0], y=[2.0], name="model_2"), row=1, col=col)
    fig = merge_saved_traces(fig, filename)

    assert [trace.name for trace in fig.data] == [
        "model_1",
        "model_1",
        "model_2",
        "model_2",
    ]
    assert [trace.xaxis for trace in fig.data] == ["x", "x2", "x", "x2"]


def test_unmatched_traces_warn(tmp_path, update_model_2):
    """
    Test a warning is raised when a saved plot has no traces to preserve.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved plot.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    filename = tmp_path / "plot.json"
    go.Figure(go.Scatter(x=[1.0], y=[1.0], name="Reference")).write_json(filename)

    with pytest.warns(UserWarning, match="No traces to preserve"):
        merge_saved_traces(go.Figure(), filename)


def test_unmatched_data_warns(tmp_path, update_model_2):
    """
    Test a warning is raised when saved data has no models to preserve.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved data.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    filename = tmp_path / "figures.json"
    with open(filename, "w") as fp:
        json.dump({"reference": {"value": 1.1}}, fp)

    with pytest.warns(UserWarning, match="No data to preserve"):
        merge_saved_models({}, filename)


def test_model_colours():
    """Test colours are assigned by model registry order, not plotted order."""
    colours = ["red", "green", "blue"]

    assert get_model_colour("model_1", colours) == "red"
    assert get_model_colour("model_2", colours) == "green"
    # Models that are not defined follow all defined models
    assert get_model_colour("model_3", colours) == "blue"
