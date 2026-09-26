=================
Adding benchmarks
=================

This guide will break down the process of adding a new benchmark into several steps:

1. :ref:`metrics`
2. :ref:`calculations`
3. :ref:`analysis`
4. :ref:`dash`

Please ensure you use the appropriate issue and pull request templates when
contributing a new benchmark. Full examples can be found by filtering by the
`example benchmark addition <https://github.com/ddmms/ml-peg/issues?q=label%3A%22example%20benchmark%20addition%22>`_
label.

A Jupyter Notebook tutorial introducing this process interactively can also be found in
the `Python tutorials <https://github.com/ddmms/ml-peg/tree/main/docs/source/tutorials/python>`_
documentation directory.

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

Each metric in ``metrics.yml`` should have a ``level_of_theory`` field identifying its reference
method. The app compares this string against the ``level_of_theory`` set in ``models.yml`` for each
MLIP, flagging mismatches using the traffic light system. An exact string match is required.

Each metric must also define ``good``, ``bad``, ``unit``, ``tooltip``, and ``weight``.
Choose the ``good`` and ``bad`` thresholds based on scientific considerations rather than the
range of scores produced by the current models. Typically, ``good`` is an error below which
further improvement offers little benefit, such as the uncertainty of the reference method.
The ``bad`` threshold marks an error at which a prediction is no longer useful and may even give
incorrect qualitative trends. It is valid for every current model to receive a low score if none
meet these criteria.

.. warning::

    Use the standard strings defined in :doc:`Levels of theory </developer_guide/levels_of_theory>`.
    A typo or inconsistent capitalisation (e.g. ``experiment`` instead of ``Experimental``) will
    cause incorrect warnings.

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

For consistency, write output files to
``ml_peg/calcs/[category]/[benchmark_name]/outputs``. Download input data at runtime rather than
committing it to the code repository.

Before implementing the calculation, check the following project conventions:

* Select the calculator precision explicitly. Static calculations, geometry optimisations, NEBs,
  and phonons normally use ``precision="high"`` and long molecular dynamics normally uses
  ``precision="low"`` unless another choice is justified.
* Before evaluating a structure, ensure ``atoms.info["charge"]`` and
  ``atoms.info["spin"]`` (spin multiplicity) are set as integers. For inputs known to be neutral
  singlets, simply assign ``atoms.info["charge"] = 0`` and ``atoms.info["spin"] = 1``. Otherwise,
  preserve the physically meaningful values supplied by the input.
* Catch failures for each independent structure, trajectory, or property, emit a warning with
  useful model/system context, and store ``NaN`` or explicit failure metadata so the remaining
  systems can still run. Bound iterative calculations and record convergence from the optimiser or
  dynamics state.
* Save raw model outputs and useful structures or trajectories under ``outputs/[model]`` in a
  standard ASE format. Retain the component quantities needed to reproduce derived observables.
  Calculate derived properties and metrics during analysis rather than in the calculation script.
* Download all input data inside the test or run function. Store it in the ML-PEG S3 bucket and use
  ``download_s3_data`` by default. Use ``download_github_data`` only when needed, and pin GitHub
  data to an immutable release or commit. See :doc:`Data </developer_guide/data>` for both helpers.
* Apply a D3 correction when it is consistent with the reference protocol.
* Mark expensive tests with ``@pytest.mark.slow`` or ``@pytest.mark.very_slow``, add progress
  reporting for long loops, and document a rough GPU runtime using a representative model, for
  example ``mace-mp-0a``, and named hardware.

Calculations and analyses should also follow the failure and mock-calculator guidance in
:doc:`Element filtering </developer_guide/filter>`.

The test contained in this file may be runnable as a standalone script,
but it should also be possible to run with ``pytest``, e.g.:

.. code-block:: bash

    pytest -v -s ml_peg/calcs/[category]/[benchmark_name]/calc_[benchmark_name].py


.. note::

    The ``-s`` stops ``pytest`` intercepting stdout, so ``print`` statements are
    printed to the console.


``pytest`` will run any functions beginning with ``test_``, enabling multiple types
of calculation to be defined, discovered, and run within the same benchmark.

Future examples will also demonstrate the use of ``fixtures``, which allow calculations such as
relaxation to be reused by multiple tests within the module.

Current examples are implemented with two alternative approaches:

a. Defining a script that iterates over models (recommended)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For consistency, we use similar model definitions as ``mlipx``. Models from
``ml_peg/models/models.yml`` can be loaded using the ``load_models`` function, while
``current_models`` allows a subset of those to be used for calculations, including
through a command-line input, ``--models``, to ``pytest``, or to ``ml_peg calc``.

Details about model definitions and loading are described in more detail in
:doc:`Adding models </developer_guide/add_models>`.

Using ``pytest`` `parametrisation <https://docs.pytest.org/en/stable/example/parametrize.html>`_,
the same calculation is run for each model name-model pair:

.. note::

    Some imports are not included in the following example for simplicity.


.. code-block:: python3

    from ml_peg.calcs.utils.utils import download_s3_data
    from ml_peg.models.get_models import load_models
    from ml_peg.models import current_models

    MODELS = load_models(current_models)
    OUT_PATH = Path(__file__).parent / "outputs"


    @pytest.mark.parametrize("mlip", MODELS.items())
    def test_benchmark(mlip: tuple[str, Any]) -> None:
        """
        Run calculations required for lithium diffusion along path B.

        Parameters
        ----------
        mlip
            Name of model use and model to get calculator.
        """
        model_name, model = mlip
        calc = model.get_calculator(precision="high")

        data_path = (
            download_s3_data(
                key="inputs/[category]/[benchmark_name]/[benchmark_name].zip",
                filename="[benchmark_name].zip",
            )
            / "[benchmark_name]"
        )
        struct = read(data_path / "struct.xyz")
        struct.info["charge"] = 0
        struct.info["spin"] = 1
        struct.calc = calc

        try:
            struct.info["pred_energy"] = struct.get_potential_energy()
        except Exception as exc:
            warn(f"Error calculating {model_name}: {exc}", stacklevel=2)
            struct.info["pred_energy"] = np.nan

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / "struct.extxyz", struct)



b. Defining a ``ZnTrack`` node to run via ``mlipx``
+++++++++++++++++++++++++++++++++++++++++++++++++++

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

    S3_KEY = "inputs/[category]/[benchmark_name]/[benchmark_name].zip"
    S3_FILENAME = "[benchmark_name].zip"

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
            calc = self.model.get_calculator(precision="high")
            data_path = (
                download_s3_data(key=S3_KEY, filename=S3_FILENAME)
                / "[benchmark_name]"
            )
            struct = read(data_path / "struct.xyz")
            struct.info["charge"] = 0
            struct.info["spin"] = 1
            struct.calc = calc

            # Run calculation
            try:
                struct.info["pred_energy"] = struct.get_potential_energy()
            except Exception as exc:
                warn(f"Error calculating {self.model_name}: {exc}", stacklevel=2)
                struct.info["pred_energy"] = np.nan

            write_dir = OUT_PATH / self.model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / "struct.extxyz", struct)


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

Analysis must tolerate models that were not run and individual calculations that failed. A failed
required result should normally propagate to the aggregate metric as ``NaN``. Do not replace
failed values with zero or silently remove them and score only the successful subset, as this
rewards a model for failing difficult systems. Reference data should be loaded once and
independently of which model happens to have an output directory.

In principle, the exact form of this is flexible, as long as the outputs can be assembled as
required in :ref:`dash` to build the new application tab.

However, we strongly recommend following the template described below, which enables automated
creation of tables and scatter plots, as well as placing structures to be visualised in an
appropriate directory to be accessed by the app.

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

This will generate a table with columns for each metric, as well as "MLIP" and "Score"
columns. Tooltips for each column header can also be set by the decorator, as well as
the location to save the JSON file to be loaded when building the app, which typically
would be placed in ``ml_peg/app/data/[category]/[benchmark_name]``.

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

All data needed by the app should be produced during analysis. App imports should not read directly
from calculation outputs or perform scientific analysis. Reuse the existing ``plot_from_*`` and
``struct_from_*`` callback helpers where possible. Keep plot values, hover labels, filenames, and
structures in the same deterministic order. Ensure a clicked model trace displays that model's
structure when geometries are model-dependent.

Absolute paths to ``ml_peg/app`` and ``ml_peg/calcs`` can be imported for
convenience.

Similarly to running calculations, we use imports from ``ml_peg.models`` to get the
model names that analysis will be run for. By default, this means all model names
defined in ``ml_peg/models/models.yml`` will be used, but when using ``pytest`` or our
CLI (``ml_peg analyse``), a subset can be used using the ``--models`` option.

In order to facilitate element filtering, it is also essential that the elemental
compositions of the systems involved in the benchmark are saved during analysis. Please
refer to :doc:`element filtering </developer_guide/filter>` for more details.

.. note::

    Some imports are not included in the following example for simplicity.


.. code-block:: python3

    from ml_peg.analysis.utils.decorators import build_table, plot_parity
    from ml_peg.analysis.utils.utils import mae
    from ml_peg.app import APP_ROOT
    from ml_peg.calcs import CALCS_ROOT
    from ml_peg.models.get_models import get_model_names
    from ml_peg.models import current_models

    MODELS = get_model_names(current_models)
    CALC_PATH = CALCS_ROOT / [category] / [benchmark_name] / "outputs"
    OUT_PATH = APP_ROOT / "data" / [category] / [benchmark_name]

    REF_VALUES = {"path_b": 0.27, "path_c": 2.5}

    INFO = get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="structs.xyz",
        index=":",
        write_info=True,
        info_keys=["label"],
        write_structs=True,
        out_path=OUT_PATH,
        include_filenames=True,
    )

    @pytest.fixture
    @plot_parity(
        filename=OUT_PATH / "figure_energies.json",
        title="Relative energies",
        x_label="Predicted energy / eV",
        y_label="Reference energy / eV",
        hoverdata={
            "Labels": INFO["label"],
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
                results["ref"] = [struct.info["ref_energy"] for struct in structs]

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

Framework credit and custom group tags
++++++++++++++++++++++++++++++++++++++

If a benchmark comes from an external benchmarking framework (for example, MLIP Audit),
or you want to group benchmarks by a custom category (such as those involved in a
paper), you can add a framework credit tag as follows:

1. Add/update the framework entry in ``ml_peg/app/utils/frameworks.yml``.

.. code-block:: yaml

    mlip_audit:
    label: MLIP Audit
    color: "#1d4ed8"
    text_color: "#ffffff"
    url: "https://huggingface.co/spaces/InstaDeepAI/mlipaudit-leaderboard"
    logo: "https://raw.githubusercontent.com/instadeepai/mlipaudit/mlpeg-migration/InstaDeep_Logo.png"

2. Set ``framework_ids`` in the benchmark app constructor. This accepts either a
   single framework identifier or a sequence of them, so a benchmark can belong to
   multiple frameworks at once.

   The app constructor also provides a ``include_ml_peg`` parameter. This is ``True``
   by default, which automatically adds the built-in ``ml_peg`` tag to benchmarks, so
   ``framework_ids`` only needs to list *additional* or *alternative* frameworks. If
   you want to omit this tag, set ``include_ml_peg=False``.

   For example:

.. code-block:: python3

    # MLIP Audit only
    return SomeBenchmarkApp(
        name="SomeBenchmark",
        ...,
        framework_ids="mlip_audit",
        include_ml_peg=False,
    )


    # MLIP Audit and the MACE Multihead paper
    return SomeBenchmarkApp(
        name="SomeBenchmark",
        ...,
        framework_ids=["mlip_audit", "mace-multihead"],
        include_ml_peg=False,
    )

    # ML-PEG (added by default) and the MACE Multihead paper
    return SomeBenchmarkApp(
        name="SomeBenchmark",
        ...,
        framework_ids="mace-multihead",
    )

That is all that is required. The benchmark header shows one badge per framework,
and the additional framework pages for non-default frameworks are populated
automatically from this metadata.

Framework sections group matching benchmarks by category, omit the category
summary table, and reuse the same benchmark tables and controls. Updating
weights or thresholds there therefore updates the same benchmark views shown in
the category pages.

The ``logo`` field is optional. It can point to a remote image URL or a local
Dash asset path such as ``/assets/frameworks/my_framework_logo.png``. Use a
browser-supported image format such as ``.svg``, ``.png``, or ``.jpg``/``.jpeg``.
``.pdf`` is not supported. For best results, use a square logo image.
