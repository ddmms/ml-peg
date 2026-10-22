Profiling the app
=================

These tools are for developers investigating app performance. They run against
an already-started app and do not perform benchmark calculations.

The browser runner times category and framework navigation, scoring edits,
table-cell clicks, model selection and element filtering using your data.

It uses Playwright, a tool that controls a real Chromium browser. The script finds
links and controls on the page, clicks sidebar links, types into input boxes and
presses Enter, much as you would manually. It follows a predefined sequence in
``ml_peg/app/utils/browser_profiling.py`` and discovers available pages from the sidebar.
Before moving on, it waits for table data, pending updates and loading overlays
to finish. The time spent performing each action and waiting is recorded.
``--headed`` means a visible browser window, so you can watch the steps.
Without it, Chromium runs headlessly, with no visible window.

The command is available when Playwright and Matplotlib are installed. In a
repository checkout, install the developer dependencies with ``uv sync --group dev``
and Chromium with ``uv run playwright install chromium``. Environments without
these Python dependencies do not list the command in ``ml_peg --help``.

Start the app, then run the profiler in another terminal:

.. code-block:: bash

    ml_peg app --no-debug
    ml_peg profile --repeats 3 --headed

- Each repetition starts with fresh browser storage and waits for updates
  and visible loading overlays to finish.
  Empty categories and unavailable benchmark interactions are skipped. Model
  selection and element filtering still run when no compatible benchmark is found.
- Results go to ``profiles/browser/<timestamp>/``: ``summary.png``,
  ``summary.txt``, ``terminal.txt`` and detailed ``results.json``.
- Use ``--help`` to select pages and benchmarks, or ``--trace`` to save a
  Playwright trace. If needed, install Chromium with
  ``python -m playwright install chromium``.
- Timings include a 300 ms quiet window and automation overhead. Compare runs
  with the same data and settings, keeping tracing off for baseline timings.

Data for a useful run
---------------------

Use analysis-generated app data with a populated summary table. For the full tour,
include at least one benchmark with an ``MAE`` column and an interactive metric
plot, at least two models for the model-selection step, and element-coverage data
for filtering. The runner uses the data served by the app, not a built-in dataset.
The UI test fixture includes IONPI19, which provides a suitable MAE plot.

Select a benchmark explicitly as below, changing ``--metric`` for other metrics:

.. code-block:: bash

    ml_peg profile --category non_covalent_interactions \
        --benchmark-category non_covalent_interactions \
        --benchmark IONPI19 --metric MAE

Category options accept either a name such as ``bulk_crystal`` or a sidebar path
such as ``/category/bulk_crystal``. Use the benchmark component name shown by the
app. Unknown sidebar paths are reported immediately rather than waiting for an
absent link. Keep the dataset and selected models fixed when comparing timings.

Python profiling
----------------

To investigate Python startup or callback costs, enable server profiling:

.. code-block:: bash

    ML_PEG_PROFILE_DIR=profiles ml_peg app --no-debug

This saves uniquely named ``startup-*.prof`` and per-request ``.prof`` files
with matching JSON timing metadata. Repeated builds retain separate startup
profiles. While profiling, debug mode keeps its diagnostics but disables the
automatic reloader to avoid duplicate profiling processes. Inspect
one with ``python -m pstats <file.prof>``, then enter ``sort cumulative`` and
``stats 30``. Server profiling adds overhead and is disabled by default.
