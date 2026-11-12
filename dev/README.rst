Developer tools
===============

Run these tools from the repository root in the project environment.

Browser profiling
-----------------

``ml_peg profile`` controls Chromium to time navigation, scoring edits and
filters.

The command is available when Playwright and Matplotlib are installed. In a
repository checkout, install the developer dependencies with ``uv sync --group dev``
and Chromium with ``uv run playwright install chromium``. Environments without
these Python dependencies do not list the command in ``ml_peg --help``.

Start the app, then run the profiler in another terminal:

.. code-block:: bash

    ml_peg app --no-debug
    ml_peg profile --repeats 3 --headed

``--headed`` shows a visible browser window. Omit it to run without a window.
Use a populated app dataset with an interactive ``MAE`` benchmark plot (for
example IONPI19) and at least two models for the full tour. Other metrics can be
selected with ``--benchmark``, ``--benchmark-category`` and ``--metric``.
Unavailable benchmark interactions are skipped, while model and element-filter
steps still run. See ``docs/source/developer_guide/profiling.rst`` for details.

Reports are saved under ``profiles/browser/``. Use ``--help`` for options.

Element coverage
----------------

- ``element_coverage/find_supported_elements.py`` probes which elements a model
  accepts using dimer energies. Run it with the relevant model backend installed.
- ``element_coverage/compare_supported_elements.py`` compares those results with
  the dataset coverage definitions used by the app.

.. code-block:: bash

    python dev/element_coverage/find_supported_elements.py --models mace-mp-0a
    python dev/element_coverage/compare_supported_elements.py

Probe results accumulate beside the scripts. Error logs are ignored by Git, while
``model_supported_elements.json`` remains available to review and commit. Accepting
an element does not establish training coverage, so review the results before
updating model metadata.
