=================
Benchmark scoring
=================


This guide will break down how benchmark metrics are scored, normalised and combined by default and how users can specify their own normalisation schemes.
Normalisation of benchmark metrics to yield a unitless score is essential if we want to combine multiple metrics with different units and orders of magnitude into a single overall benchmark score.


Weights and Score Aggregation
-----------------------------

The scoring system uses a 5-level hierarchy:

- **Raw metric values**: Each benchmark contains one or more metrics (e.g., energy MAE, phonon max and min frequency MAE, etc.), which are raw values with units.
- **Normalised metric scores**: Each raw metric value is normalised to a unitless score using "good" and "bad" thresholds. These scores can be viewed with the "Show normalised scores" toggle.
- **Benchmark scores**: After normalisation, benchmark scores are calculated as a weighted average of the normalised metric scores within that benchmark.
- **Category scores**: Category scores are calculated as a weighted average of the benchmark scores within that category.
- **Overall scores**: Overall scores are calculated as a weighted average of the category scores.

In general, by default, metrics are equally weighted. However, users can specify custom weights by editing the text box below the corresponding column.

For developers to specify custom default weights, you can do so in the benchmark's ``metrics.yml`` configuration file by adding a ``weight`` field for each metric, e.g.:

.. code-block:: yaml

   metrics:
     MAE:
       good: 0.0
       bad: 100.0
       weight: 2.0
       unit: kJ/mol
       tooltip: "Mean Absolute Error for all systems"
       level_of_theory: DMC


See :doc:`Levels of theory </developer_guide/levels_of_theory>` for the full list of accepted
``level_of_theory`` strings.

These weights are returned as part of the tuple from ``load_metrics_config``

.. code-block:: python

   DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(METRICS_CONFIG_PATH)


and should be passed to the ``@build_table`` decorator in the analysis file, e.g.:

.. code-block:: python

   @build_table(
       filename=OUT_PATH / "x23_metrics_table.json",
       metric_tooltips=DEFAULT_TOOLTIPS,
       thresholds=DEFAULT_THRESHOLDS,
       mlip_name_map=D3_MODEL_NAMES,
       weights=DEFAULT_WEIGHTS,
   )

Benchmark and category weights
------------------------------

Default weights for whole benchmarks and whole categories are set in the category's
``ml_peg/app/[category]/[category].yml`` file. ``weight`` sets the category's weight in
the overall score, and ``benchmark_weights`` sets the weight of individual benchmarks
within that category's score. Benchmarks that are not listed default to a weight of 1:

.. code-block:: yaml

   title: NEBs
   description: Nudged Elastic Band comparisons
   weight: 1
   benchmark_weights:
     OC20NEB: 0
     grambow_organics: 0

Keys under ``benchmark_weights`` must match the benchmark's directory name, and appending
``" Score"`` to form the summary table column is handled by the app.

A weight of 0 excludes the column from the score entirely: it contributes to neither the
weighted sum nor the sum of weights, so it does not drag the score down, and the column is
dimmed in the table. This is not the same as scoring 0. If every benchmark in a category is
weighted 0, that category has no score.

Weights set this way are defaults only. Users can still change them in the app, using the
weight boxes below each column, or the weight presets in the sidebar.

Slow benchmarks are weighted 0 by default
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Benchmarks whose calculations are marked ``@pytest.mark.slow``, ``@pytest.mark.very_slow``
or ``@pytest.mark.multi_day`` are excluded from default scores, since results for them are
sparse and expensive to reproduce across models. Users who want them can re-weight them in
the app.

This is a manual mirror of the pytest speed markers, so when adding a slow benchmark, or
changing an existing benchmark's speed marker, remember to update ``benchmark_weights``
to match. ``tests/test_speed_markers.py::test_slow_benchmarks_are_zero_weighted`` enforces
this, and will fail with the entry that needs adding. The reverse is not enforced: a fast
benchmark may be weighted 0 for its own reasons.

Missing results
---------------

Results are missing for all sorts of reasons: a benchmark has not been run yet, a model
crashed on one system, a model is absent from a benchmark entirely, or an element filter
removed every system a benchmark covers. Cells show this as either ``None`` or ``NaN``,
and which one a cell holds reflects where it came from, not anything about the model.

Scoring therefore treats them identically, and decides **per column, not per row**, so
that a score always means the same thing for every model
(``calc_table_scores``, ``ml_peg/analysis/utils/utils.py``):

- A column weighted 0 is excluded for every model. Missing values in it are harmless.
- A column holding a value for **at least one** model is scored for **every** model. Any
  model missing a value there scores ``NaN``, and that ``NaN`` propagates up the
  hierarchy, so the model has no benchmark, category or overall score.
- A column holding a value for **no** model is discarded for everyone, exactly as a
  weight of 0 would be. It therefore does not stop anyone reaching an overall score.
- If no column survives, the score is ``None`` rather than ``NaN``: nothing was scored,
  as opposed to something being wrong.

The guarantee this buys is that, for a given set of weights and element filters, every
model's score at every level is the weighted average of exactly the same combination of
metrics. Scores stay comparable between models, which is the point of the leaderboard.

.. warning::

   Do not reintroduce a per-row rule such as "skip a missing value and score the rest".
   It silently averages different models over different metrics, and lets a weighted
   column vanish from one model's score while counting in another's.

``build_summary_table`` (``ml_peg/app/utils/build_components.py``) applies the same rule to
the weights themselves: a summarised benchmark or category that scores nothing for anyone
has its default weight overridden to 0, so it is visibly weighted 0 and says so in its
header tooltip, rather than being silently ignored.

A column dropped by an element filter is not treated this way, because overwriting the
stored weight would discard the user's own setting once the filter is lifted. Instead
``get_table_style`` dims any column holding no value for any model, exactly as it dims a
zero-weighted one, so a column that stops counting always looks switched off whether it
was weighted out, never run, or filtered away. Dimming is display only and reverses with
the filter.

Scoring and normalisation
-------------------------

By default, every benchmark metric is normalised to a unitless score between 0 (worst) and 1 (best) using "good" and "bad" thresholds specified in each benchmark's metric configuration file (``metrics.yml``).
In the app, thresholds can be customised for each metric by users to suit their own requirements, by editing the text boxes below the corresponding columns. For example, a phonon expert may have strict requirements for phonon frequency accuracy, whereas a user interested in general trends may be satisfied with a more approximate model.

.. figure:: ../_images/normalization_scheme.png
   :alt: ML-PEG metric normalisation scheme
   :align: center
   :width: 100%

The **"good" threshold** defines a raw metric value where performance better than this value is considered as good as a user would require. Further improvements in performance beyond this threshold are unlikely to be impactful.

The **"bad" threshold** defines a raw metric value where performance worse than this value indicates that the model should be avoided for this task.

Default values for thresholds have been carefully chosen for each metric to help users decide in which situations foundation models perform well and are reliable to use, and in which situations they perform poorly and should be treated with caution or avoided completely.

The figure above illustrates the default normalisation scheme used. Here, the normalisation is linear between these thresholds, with scores capped at 0 and 1 outside the thresholds. For example, if a metric has a "good" threshold of 1.0 kcal/mol and a "bad" threshold of 5.0 kcal/mol, a raw metric value of 1.0 kcal/mol or lower will yield a score of 1.0, a raw value of 5.0 kcal/mol or higher will yield a score of 0.0, and a raw value of 3.0 kcal/mol will yield a score of 0.5.



Advanced users can also define their own custom normalisation functions by defining a new ``normalizer`` in ``ml_peg/analysis/utils/utils.py``.
