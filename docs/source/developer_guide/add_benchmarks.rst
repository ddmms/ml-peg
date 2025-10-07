=================
Adding benchmarks
=================

This guide will break down the process of adding a new benchmark into several steps:

1. :ref:`metrics`
2. :ref:`calculations`
3. :ref:`analysis`
4. :ref:`dash`

.. _metrics:

Identifying metrics
-------------------

Having selected an application/system/property of interest to test,
the first step is to identify quantifiable metrics of performance.

Often, this will be a comparison to a reference value, such as
the error with respect to DFT predictions of energies.

.. note::

    In future, we expect to support user-selection of error metrics,
    e.g. MAE, RMSE, or higher-order errors.


Reference data may also include experimental data, or higher-accuracy theoretical predictions,
e.g. CCSD(T).

In some cases, metrics may also encode correct behaviour without a specific reference,
such as by quantifying features of a known distribution (curvature, minima, etc.),
or quantifying the stability of a simulation.


.. _calculations:

Running Calculations
--------------------

1. Create a new directory in ``ml_peg/calcs/[category]`` with a short, unique benchmark name.

2. Write a script that will run the MLIP calculations of interest for each model being tested.

The file should be named ``calc_[benchmark_name].py``,
and placed in ``ml_peg/calcs/[category]/[benchmark_name]``.

While not a requirement, we recommend placing input files in
``ml_peg/calcs/[category]/[benchmark_name]/data``, and output files in
``ml_peg/calcs/[category]/[benchmark_name]/outputs``, for consistency.

The test contained in this file may be runnable as a standalone script,
but it should also be possible to run with ``pytest``, e.g.:

.. code-block:: bash

    pytest -v -s ml_peg/calcs/[category]/[benchmark_name]/calc_[benchmark_name].py


.. note::

    The ``-s`` stops ``pytest`` intercepting stdout, so ``print`` statements will
    be print to the console.


``pytest`` will run any functions beginning with ``test_``, enabling multiple types
of calculation to be defined, discovered, and run within the same benchmark.

Future examples will also demonstrate the use of ``fixtures``, which allow calculations such as
relaxation to be reused by multiple tests within the module.

Current examples are implemented with two alternative approaches:

a. Defining a script that iterates over models:

For consistency, we use similar model definitions as ``mlipx``. Models from
``ml_peg/models/models.yml`` can be loaded using the
``ml_peg.models.get_models.load_models`` function, while
``ml_peg.models.models.current_models`` allows a subset of those to be used for
calculations, including through a command-line input, ``--models``, to ``pytest``, or
to ``ml_peg calc``.

Details about model definitions and loading are described in more detail in
:doc:`Adding models </developer_guide/add_models>`.

Using ``pytest`` `parametrisation <https://docs.pytest.org/en/stable/example/parametrize.html>`_,
the same calculation is run for each model name-model pair:

.. note::

    Some imports are not included in the following example for simplicity.


.. code-block:: python3

    from ml_peg.models.get_models import load_models
    from ml_peg.models.models import current_models

    MODELS = load_models(current_models)
    DATA_PATH = Path(__file__).parent / "data"
    OUT_PATH = Path(__file__).parent / "outputs"


    @pytest.mark.parametrize("mlip", MODELS.items())
    def test_benchmark(mlip: tuple[str, GenericASECalculator]) -> None:
        """
        Run calculations required for lithium diffusion along path B.

        Parameters
        ----------
        mlip
            Name of model use and model to get calculator.
        """
        model_name, model = mlip

        struct = read(DATA_PATH / "struct.xyz")
        struct.calc = model.get_calculator()

        struct.get_potential_energy()

        write_dir = OUT_PATH / self.model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / "struct.xyz", struct)



b. Defining a ``ZnTrack`` node to run via ``mlipx``:

The process of running these is largely as
`described by mlipx <https://mlipx.readthedocs.io/en/latest/quickstart/cli.html>`_,
including running ``dvc init`` in ``ml_peg/calcs/[category]/[benchmark_name]``.

.. note::

    In general, this would also require running ``git init``,
    but the repository should already be tracked by git.


In this example, we create the ``NewBenchmark`` node,
which defines a ``run`` function to perform the calculation using each model,
which ``mlipx`` automatically sets via the zntrack.deps().

``mlipx`` also sets ``model_name`` via ``zntrack.params()``, which
we use to differentiate the output files.

We also define ``test_new_benchmark``, which enables this benchmark to be automatically
run identified and run using ``pytest``.

.. note::

    Some imports are not included in the following example for simplicity.


.. code-block:: python3

    # Local directory to store input data
    DATA_PATH = Path(__file__).parent / "data"

    # Local directory to store output data
    OUT_PATH = Path(__file__).parent / "outputs"

    # New benchmark node
    class NewBenchmark(zntrack.Node):
        """New benchmark."""

        model: NodeWithCalculator = zntrack.deps()
        model_name: str = zntrack.params()

        def run(self):
            """Run new benchmark."""
            # Read in data and attach calculator
            calc = self.model.get_calculator()
            struct = read(DATA_PATH / "struct.xyz")
            struct.calc = calc

            # Run calculation
            struct.get_potential_energy()

            write_dir = OUT_PATH / self.model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / "struct.xyz", struct)


    def build_project(repro: bool = False) -> None:
        """
        Build mlipx project.

        Parameters
        ----------
        repro
            Whether to call dvc repro -f after building.
        """
        project = mlipx.Project()
        benchmark_node_dict = {}

        for model_name, model in MODELS.items():
            with project.group(model_name):
                benchmark = NewBenchmark(
                    model=model,
                    model_name=model_name,
                )
                benchmark_node_dict[model_name] = benchmark

        if repro:
            with chdir(Path(__file__).parent):
                project.repro(build=True, force=True)
        else:
            project.build()


    def test_new_benchmark():
        """Run new benchmark via pytest."""
        build_project(repro=True)


.. _analysis:

Analysing Calculations
----------------------

The output files created by :ref:`calculations` must then be analysed to calculate the metrics
as planned in :ref:`metrics`.

In principle, the exact form of this flexible, as long as the outputs can be assembled as required
in :ref:`dash` to build the new application tab.

However, we strongly recommend following the template described below, which enables automated
creation of tables and scatter plots, as well as placing structures to be visualised in an appropriate
directory to be accessed by the app.

As with the script created in :ref:`calculations`, we create a new file to be run by ``pytest``,
containing a function beginning with ``test_`` to launch the analysis.

In this case, we name the file
``ml_peg/analysis/[category]/[benchmark_name]/analyse_[benchmark_name].py``,
such that it can be run using:

.. code-block:: bash

    pytest -v -s ml_peg/analysis/[category]/[benchmark_name]/analyse_[benchmark_name].py


In order to automatically generate the components for our application, we will make use
of decorators, such as ``@build_table`` and ``@plot_parity``, which use the value
returned by the function, in combination with any parameters set for the decorator.
This therefore requires the values returned by decorated functions to be of a
particular form.

For ``@build_table``, the value returned should be of the form:

.. code-block:: python3

    {
        "metric_1": {"model_1": value_1, "model_2": value_2, ...},
        "metric_2": {"model_1": value_3, "model_2": value_4, ...},
        ...
    }

This will generate a table with columns for each metric, as well as "MLIP", "Score",
and "Rank" columns. Tooltips for each column header can also be set by the decorator,
as well as the location to save the JSON file to be loaded when building the app,
which typically would be placed in ``ml_peg/app/data/[category]/[benchmark_name]``.

Every benchmark should have at least one of these tables, which includes
the score for each metric, and allowing the table to calculate an overall score for the
benchmark, and so often this decorated function is called as a fixture by the ``test_``
function.

Benchmarks may also include other tables, which can be built similarly, although
currently the scores from these cannot be straightforwardly combined into an overall
table.

For ``@plot_parity``, the value returned should be of the form:

.. code-block:: python3

    {
        "ref": ref_values_list,
        "model_1": model_1_values_list,
        "model_2": model_2_values_list,
        ...
    }


This will generate a scatter plot of reference value against model value for each model,
as well as a dashed line representing ``y=x``. Additional options can be set to specify
the plot title, axes labels, and hover data.

Hover data will always include x and y values, but additional labels for each point are set
via a dictionary of label names and lists of labels (corresponding to the same data points as
``ref_values_list`` etc.):

.. code-block:: python3

    {
        "label_1": label_1_list,
        "label_2": label_2_list,
        ...
    }


Typically, functions like this that generate plots would also be fixtures that are passed to
another function, which performs the aggregation needed to then pass the metric's value
to the function that generates the table for all metrics.

Further decorators will be added as required for common figures, including bar charts,
and non-parity scatter plots.

While not essential, we can also make use of the ``@pytest.fixture`` decorator,
which allows the value returned by a function to be used directly as a parameter
for other functions.

If your benchmark contains structures to be visualised, or images to be loaded, these
should be saved to ``ml_peg/app/data/[category]/[benchmark_name]``, as they must
be added as ``assets`` to be loaded into the app.

Absolute paths to ``ml_peg/app`` and ``ml_peg/calcs`` can be imported for
convenience.

Similarly to running calculations, we use imports from ``ml_peg.models`` to get the
model names that analysis will be run for. By default, this means all model names
defined in ``ml_peg/models/models.yml`` will be used, but when using ``pytest`` or our
CLI (``ml_peg analyse``), a subset can be used using the ``--models`` option.

.. note::

    Some imports are not included in the following example for simplicity.


.. code-block:: python3

    from ml_peg.analysis.utils.decorators import build_table, plot_parity
    from ml_peg.analysis.utils.utils import mae
    from ml_peg.app import APP_ROOT
    from ml_peg.calcs import CALCS_ROOT
    from ml_peg.models.get_models import get_model_names
    from ml_peg.models.models import current_models

    MODELS = get_model_names(current_models)
    CALC_PATH = CALCS_ROOT / [category] / [benchmark_name] / "outputs"
    OUT_PATH = APP_ROOT / "data" / [category] / [benchmark_name]

    REF_VALUES = {"path_b": 0.27, "path_c": 2.5}

    def labels() -> list:
        """
        Get list of labels.

        Returns
        -------
        list
            List of all energy labels.
        """
        structs = read(CALC_PATH / "structs.xyz", index=":")
        return [struct.info["label"] for struct in structs]


    @pytest.fixture
    @plot_parity(
        filename=OUT_PATH / "figure_energies.json",
        title="Relative energies",
        x_label="Predicted energy / eV",
        y_label="Reference energy / eV",
        hoverdata={
            "Labels": labels(),
        },
    )
    def energies() -> dict[str, list]:
        """
        Get energies for all structures.

        Returns
        -------
        dict[str, list]
            Dictionary of all reference and predicted relative energies.
        """
        results = {"ref": []} | {mlip: [] for mlip in MODELS}
        ref_stored = False
        for model_name in MODELS:
            structs = read(CALC_PATH / model_name / "structs.xyz", index=":")

            results[model_name] = [struct.get_potential_energy() for struct in structs]

            if not ref_stored:
                results["ref"] [struct.info["ref_energy"] for struct in structs]

                # Write structures for app
                structs_dir = OUT_PATH / model_name
                structs_dir.mkdir(parents=True, exist_ok=True)
                write(structs_dir / "structs.xyz", structs)
            ref_stored = True

        return results


    @pytest.fixture
    def metric_1(energies: dict[str, list]) -> dict[str, float]:
        """
        Get metric 1.

        Parameters
        ----------
        energies
            Reference and predicted energies for all structures.

        Returns
        -------
        dict[str, float]
            Dictionary of metric 1 values for each model.
        """
        results = {}
        for model_name in MODELS:
            results[model_name] = mae(energies["ref"], energies[model_name])

        return results


    @pytest.fixture
    def metric_2() -> dict[str, float]:
        """
        Get metric 2.

        Returns
        -------
        dict[str, float]
            Dictionary of metric 2 values for each model.
        """
        results = {}
        for model_name in MODELS:
            structs = read(CALC_PATH / model_name / "structs.xyz", index=":")
            results[model_name] = mae(
                pred_properties, [struct.info["property"] for struct in structs]
            )

        return results


    @pytest.fixture
    @build_table(
        filename=OUT_PATH / "new_benchmark_metrics_table.json",
        metric_tooltips={
            "Model": "Name of the model",
            "Metric 1": "Description for metric 1 (units)",
            "Metric 2": "Description for metric 2 (units)",
        },
    )
    def metrics(
        metric_1: dict[str, float], metric_2: dict[str, float]
    ) -> dict[str, dict]:
        """
        Get all new benchmark metrics.

        Parameters
        ----------
        metric_1
            Metric 1 value for all models.
        metric_2
            Metric 2 value for all models.

        Returns
        -------
        dict[str, dict]
            Metric names and values for all models.
        """
        return {
            "Metric 1": metric_1,
            "Metric 2": metric_2,
        }


    def test_new_benchmark(metrics: dict[str, dict]) -> None:
        """
        Run new benchmark analysis.

        Parameters
        ----------
        metrics
            All new benchmark metric names and dictionary of values for each model.
        """
        return


.. _dash:

Build Dash components
---------------------

Any tables and figures to be added to the app should have been created and saved by
running the test defined in :ref:`analysis`.

The final step is to assemble these, by defining a ``layout``, and set up any required
interactivity, by defining ``callback`` functions, for the Dash application.

Building those components and their interactivity should become increasingly automated,
but less standard plots/interactions will need setting up.

For now, please contact us to help with this process.
