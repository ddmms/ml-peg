Profiling the app
=================

Browser profiling
-----------------

The browser runner uses Playwright to time navigation, scoring edits, table
interactions, model selection and element filtering against a running app.

Install the developer dependencies and Chromium:

.. code-block:: bash

    uv sync --group dev
    uv run playwright install chromium

Start the app, then run the profiler in another terminal:

.. code-block:: bash

    ml_peg app --no-debug
    ml_peg profile --repeats 3 --headed

The command is available when Playwright and Matplotlib are installed.
``--headed`` shows the browser window. Omit it to run headlessly.

Use populated app data. A full run needs a benchmark with an interactive metric
plot, at least two models and element-coverage data. Unavailable interactions
are skipped. To select a benchmark explicitly:

.. code-block:: bash

    ml_peg profile --category non_covalent_interactions \
        --benchmark-category non_covalent_interactions \
        --benchmark IONPI19 --metric MAE

Results go to ``profiles/browser/<timestamp>/`` as ``summary.png``,
``summary.txt``, ``terminal.txt`` and detailed ``results.json``.
Use ``--help`` for options or ``--trace`` to save a Playwright trace.

Timings include a 300 ms quiet window and automation overhead. Compare runs
with the same data and settings, keeping tracing off for baseline timings.

Python profiling
----------------

To profile Python startup and callbacks:

.. code-block:: bash

    ML_PEG_PROFILE_DIR=profiles ml_peg app --no-debug

This saves unique ``startup-*.prof`` and per-request ``.prof`` files with JSON
timing metadata. Inspect a profile with ``python -m pstats <file.prof>``, then
enter ``sort cumulative`` and ``stats 30``. Profiling adds overhead and disables
the automatic reloader.
