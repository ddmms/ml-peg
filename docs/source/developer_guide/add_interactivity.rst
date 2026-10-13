========================
Adding new interactivity
========================

When changing app controls or callbacks, add regression tests for the affected
behaviour. Check exact benchmark, category, framework and overall scores, not
just whether a value changes. Check that edits survive navigation and interact
correctly with model and element filters.

Testing
-------

Run these commands from the repository root. Install Chromium once:

.. code-block:: bash

   uv run playwright install chromium

Run the numerical scoring tests and all browser tests:

.. code-block:: bash

   uv run pytest tests/test_scoring_contract.py tests/app -q

For only the focused scoring tests, which use a small synthetic dataset:

.. code-block:: bash

   uv run pytest tests/test_scoring_contract.py tests/app/scoring -q

To watch the browser, or retain traces and screenshots for failures:

.. code-block:: bash

   uv run pytest tests/app/scoring --headed -q
   uv run pytest tests/app --tracing=retain-on-failure --screenshot=only-on-failure --output=test-results -q

The full browser suite downloads a pinned test fixture when needed and keeps
test data separate from local analysis outputs. Tests marked ``xfail`` document
known bugs, not passing behaviour. Use ``-rx`` to see their reasons.
