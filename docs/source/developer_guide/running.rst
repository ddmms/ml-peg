=============
Running tests
=============

This guide will break down how to run calculations, analysis, and the interactive
application.


Calculations
------------

All calculations can be launched using our ``ml_peg calc`` command-line command.

Help for this command can be found by running ``ml_peg calc --help``:

.. code-block:: bash

    Usage: ml_peg calc [OPTIONS]

    Run calculations

    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --models                                 TEXT  Comma-separated models to run calculations on. Default is all models.      │
    │ --category                               TEXT  Category to run calculations for. Default is all categories. [default: *]  │
    │ --test                                   TEXT  Test to run calculations for. Default is all tests. [default: *]           │
    │ --run-slow         --no-run-slow               Whether to run calculations labelled slow. [default: run-slow]             │
    │ --run-very-slow    --no-run-very-slow          Whether to run calculations labelled very slow.                            │
    │                                                [default: no-run-very-slow]                                                │
    │ --verbose          --no-verbose                Whether to run pytest with verbose and stdout printed. [default: verbose]  │
    │ --help                                         Show this message and exit.                                                │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


``ml_peg calc`` launches calculations using ``pytest``, and will automatically
discover and run each test, handle intermediate errors, and control which tests are
run based on our
`custom markers <https://docs.pytest.org/en/7.1.x/example/markers.html>`_.

For example, to run the ``S24`` test in the ``surfaces`` category, with the
``mace-mp-0b3`` model, you could run:

.. code-block:: bash

    ml_peg calc --category surfaces --test S24 --models mace-mp-0b3


This is effectively equivalent to:

.. code-block:: bash

    .. code-block:: bash

    pytest -vvv ml_peg/calcs/surfaces/S24/calc_S24.py --models mace-mp-0b3


Analysis
--------

Similarly to calculations, analysis can be launched using our ``ml_peg analyse``
command-line command.

Help for this command can be found by running ``ml_peg analyse --help``:

.. code-block:: bash

    Usage: ml_peg analyse [OPTIONS]

    Run calculations

    ╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --models                      TEXT  Comma-separated models to run analysis for. Default is all models.         │
    │ --category                    TEXT  Category to run analysis for. Default is all categories. [default: *]      │
    │ --test                        TEXT  Test to run analysis for. Default is all tests. [default: *]               │
    │ --update      --no-update           Whether to update saved tables and plots, preserving results for models    │
    │                                     not being analysed, rather than overwriting them. [default: no-update]     │
    │ --verbose     --no-verbose          Whether to run pytest with verbose and stdout printed. [default: verbose]  │
    │ --help                              Show this message and exit.                                                │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


``ml_peg analyse`` launches analysis using ``pytest``.

For example, to run the ``OC157`` test in the
``surfaces`` category, with the ``mace-mp-0b3`` and ``orb-v3-consv-inf-omat``
models, you could run:

.. code-block:: bash

    ml_peg analyse --category surfaces --test OC157 --models mace-mp-0b3,orb-v3-consv-inf-omat


This is effectively equivalent to:

.. code-block:: bash

    .. code-block:: bash

    pytest -vvv ml_peg/analysis/surfaces/OC157/analyse_OC157.py --models mace-mp-0b3,orb-v3-consv-inf-omat


Adding a model to existing analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, running analysis for a subset of models rebuilds each benchmark's table
from scratch, so models that were not analysed lose their saved metric values.

To add a new model without rerunning analysis for every other model, use
``--update``:

.. code-block:: bash

    ml_peg analyse --category surfaces --test OC157 --models my-new-model --update

Results for models outside ``--models`` are then taken from the saved table and plots,
while results for the analysed models are rebuilt from the current run. This applies to
tables built with ``@build_table``, and to plots that show a trace per model, such as
``@plot_parity``.

Scores, weights, thresholds, and tooltips are recalculated for every row, so changes to
a benchmark's thresholds are still applied to preserved rows. Axis limits and parity
lines are likewise recalculated across preserved and new traces.

.. note::

    Results are matched to models by name. A preserved row will have empty values for
    any metric that has been renamed or added since it was last analysed, and results
    are only preserved for models that are still defined in ``models.yml``.


Updating bespoke figures
~~~~~~~~~~~~~~~~~~~~~~~~

Benchmarks that build their own multi-model figures, rather than using one of the
decorators above, must opt in to updating. Two helpers in
``ml_peg/analysis/utils/decorators.py`` do this, and both return their input unchanged
when ``--update`` is not set:

``merge_saved_traces(fig, filename)``
   Adds traces for models that are not being analysed from the figure saved at
   ``filename``. Call it immediately before writing the figure. Traces are matched to
   models by ``name``, and are reordered to match ``models.yml``. Traces without a
   model name, such as reference curves, are rebuilt from the current run. Figures
   built with ``make_subplots`` are supported, as preserved traces keep their axis
   assignment.

``merge_saved_models(model_data, filename, key=None)``
   Adds entries for models that are not being analysed to a mapping of model name to
   data, such as a figure per model saved in a single file. Pass ``key`` if the saved
   file holds that mapping under a key, rather than being keyed by model itself.

Figures that colour traces per model must also assign colours with
``get_model_colour(model, colours)``, which indexes ``colours`` by the model's position
in ``models.yml``. Indexing by plotted order instead gives the analysed models the same
colours as the preserved traces.

A warning is raised when a saved file exists but contains nothing to preserve, which
usually means its traces or keys are not named after models.

.. warning::

    Updating reads, modifies, and rewrites each saved file, so analysis for a single
    benchmark must not be updated by several runs at once. To add multiple models,
    pass them together as ``--models model_1,model_2``, or run each update in turn.


Application
-----------

Having run analysis, the app can now be launched by running the ``ml_peg app``
command-line command.

Help for this command can be found by running ``ml_peg app --help``:

.. code-block:: bash

    Usage: ml_peg app [OPTIONS]

    Run application

    ╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --models                    TEXT  Comma-separated models to build interactivity for. Default is all models. │
    │ --category                  TEXT  Category to build app for. Default is all categories. [default: *]        │
    │ --port                      TEXT  Port to run application on. [default: 8050]                               │
    │ --debug       --no-debug          Whether to run with Dash debugging. [default: debug]                      │
    │ --help                            Show this message and exit.                                               │
    ╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

.. note::

    The ``models`` option for this command only influences building interactive
    callbacks, and does not change whether the models are included in tables, scores,
    or summaries,

When launched, the app will attempt to automatically construct tables, figures, and
interactive features, based on any importable test apps defined in ``ml_peg/apps/``.

If any plots are unable to be loaded, a warning will be raised, and only the table will
be rendered for the test.

If a test's table is also unable to be loaded, the test will not be added to the app,
but the app builder should continue to attempt adding other tests.

By default, the live app can then be accessed at http://localhost:8050.

To run the app on a different port (e.g. 8060), and for only the NEBs category, run:

.. code-block:: bash

    ml_peg app --category nebs --port 8060
