Developer tools
===============

Run these tools from the repository root in the project environment.

Browser profiling
-----------------

``profile_app.py`` controls Chromium to time navigation, scoring edits and
filters. Start the app, then run the profiler in another terminal:

.. code-block:: bash

    ml_peg app --no-debug
    python dev/profile_app.py --repeats 3 --headed

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

Probe results accumulate beside the scripts and are ignored by Git. Accepting
an element does not establish training coverage, so review the results before
updating model metadata.
