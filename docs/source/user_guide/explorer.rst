==============
Model Explorer
==============

Open **Explorer** in the sidebar (``/explorer``) to shortlist models, compare shared
observations and inspect individual errors. The page reads the app's shipped metrics
tables, parity figures and XYZ files at startup. The page header shows scored benchmark
coverage and the number of observations with comparable errors.

Only parity points with identifiable observations, aligned arrays and a common reference
contribute to structure comparisons. Some benchmarks contribute scores only. For example,
OC157's current relative-energy figure identifies compositions shared by multiple systems.
Geometry links also depend on an unambiguous match to a shipped XYZ file.

The page is split into seven panels, described below in the order they appear. Each
panel has a collapsed *How this works* note in the app, and a *Methods & references*
panel at the bottom cites the source for each method.


Scores and errors used by the Explorer
--------------------------------------

The Explorer uses two kinds of evidence:

- **Benchmark scores** are the app's usual 0 (worst) to 1 (best) scores, one per model
  per benchmark, normalised using each benchmark's "good" and "bad" thresholds. They
  are a *relative grade* against each benchmark's own target, not an absolute error.
  See :doc:`Benchmark scoring </developer_guide/scoring_and_normalisation>` for how these
  are computed.
- **Per-structure errors** are reconstructed from each benchmark's parity plot: for
  every structure (or reaction, conformer, ...) the absolute difference between a
  model's prediction and the reference. Because units differ between benchmarks, these
  are only ever compared *within* a benchmark, or normalised first (by rank, or by the
  benchmark's median error) before being combined.

Physicality benchmarks (locality, extensivity, diatomic curves, ...) test whether a
model behaves sensibly rather than how accurate it is, so the Explorer reports them
separately as *soundness* and never folds them into an accuracy score. Benchmarks that
measure cost rather than accuracy, such as inference speed, are left out.


In-domain versus extrapolation
------------------------------

Each model in ``models.yml`` records the level of theory it was trained on, and each
benchmark metric records the level of theory of its reference data (see
:doc:`Levels of theory </developer_guide/levels_of_theory>`). The Explorer uses the same
comparison as the warning badges in the benchmark tables to label every
model--benchmark pairing:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Label
     - Meaning
   * - in-domain
     - The benchmark reference matches the model's training level of theory.
   * - functional mismatch
     - Both are DFT, but with different functionals (e.g. a PBE-trained model on an
       r2SCAN benchmark).
   * - higher-level reference
     - The reference is a higher-level method (coupled cluster, DMC, ...) than the
       DFT the model was trained on.
   * - vs experiment
     - The reference is experimental.
   * - unclassified
     - The reference or model training level is unavailable.

A benchmark with several metrics takes the most severe known mismatch among them.
A match describes the recorded levels of theory. It does not establish that the
benchmark chemistry was in the training set or predict how a model will perform on
your system.


Model finder
------------

Describe what you are simulating and get a ranked shortlist of models with the evidence
behind each pick.

Controls
^^^^^^^^

- **What are you simulating?** — an experiment preset that selects a bundle of
  benchmark areas: *Anything*, *Bulk materials & crystals*, *Surfaces, defects &
  reactions*, *Molecules & non-covalent chemistry*, or *Metals & f-block chemistry*.
- **Elements (match any selected)** — when set, the Finder adds evidence from
  observations containing at least one selected element.
- **Benchmark areas (fine-tune)** — edit the category list directly; this switches the
  preset to custom.
- **Evidence to trust** — the extrapolation tolerance. *In-domain evidence only* counts
  only benchmarks whose reference matches the model's training level of theory;
  *Allow DFT-functional mismatch* also admits benchmarks against a different DFT
  functional; *Use all evidence* admits everything.

Reading the results
^^^^^^^^^^^^^^^^^^^

The bar chart ranks the top models by *fit*. Below it, evidence cards for the top three
picks state the model's training level of theory, its mean benchmark score with the
number of benchmarks and the share that were in-domain, how it ranks against other
models on structures containing your elements, and a physicality check. Clicking a bar
or a card opens that model's report card.

How the ranking works
^^^^^^^^^^^^^^^^^^^^^

Each model's fit combines two independent evidence streams:

1. **Benchmark evidence**: the model's mean 0--1 score over the selected areas, counting
   only benchmarks that pass the chosen extrapolation tolerance.
2. **Structure evidence** (only when elements are selected): the Finder ranks models by
   absolute error for each matching observation. Only models with a score for that
   benchmark that passes the selected tolerance enter its comparison. A tie counts as
   half a win. The mean win rate gives each benchmark equal weight and depends on the
   set of models evaluated.

The fit is then

.. math::

   \text{fit} = 0.6 \times \text{benchmark} + 0.4 \times \text{structure}

When no elements are selected or a model has no eligible structure comparisons, its
fit uses benchmark evidence alone. Coverage affects the benchmark evidence in two ways:

- A model needs at least **3** admissible benchmarks to be ranked at all.
- Each model's benchmark mean is shrunk towards the pooled mean over all candidates
  (empirical-Bayes shrinkage with a prior strength of **4** pseudo-observations), so a
  model that aced a handful of benchmarks cannot outrank one that is solid across many.
  Hover over a bar to see the benchmark count behind it.

Physicality is checked for every candidate regardless of the areas selected; a mean
physicality score below **0.5** is flagged on the card as *may behave unphysically*.

.. tip::

    Start with *Use all evidence* to see the whole field, then tighten the tolerance to
    *In-domain evidence only* to see which picks survive when only directly comparable
    evidence counts.


Model report card
-----------------

Pick one model to see its mean score in every benchmark category, plotted against the
all-model mean (grey tick) with the model's rank in that category. Categories are sorted
worst to best, so the weak spots are at the top. Each bar is coloured by whether the
model is mostly tested in-domain or by which kind of extrapolation; hover for the exact
breakdown. Physicality is summarised separately.

The table beneath lists every benchmark the model was scored on, with its score,
reference level of theory and in-domain/extrapolation label. Category names link to the
corresponding category page, so you can go from "this model is weak on surfaces" to the
individual surface benchmarks in one click.

.. note::

    Ranks are relative to whichever models were run in that category, and the cohort
    size varies between categories. Treat close ranks as ties.


Head-to-head
------------

Compare two models on observations with a reference and an error for both models. A win
means the model had the smaller absolute error. The chart shows one bar per category.

- **Direction** points toward the model with the higher win rate.
- **Length** shows the margin from 50/50.
- **Grey bars** indicate a margin within five percentage points of an even split.
- **Hover** shows comparison counts, including exact ties.

Calculate the win rate within each metric, average the rates within each benchmark,
then average benchmarks within each category. Ties count toward coverage and are
excluded from the rate. A category with all ties appears even.

Select a bar or an area to see individual comparisons. Axes show each model's absolute
error divided by the metric's median absolute error, using the median positive error
if the overall median is zero. Zero errors sit at the plot's display floor and appear
as zero on hover. Click a point to see its errors and any matching 3D geometry.


Where models fail
-----------------

The structures the field gets most wrong. With no model selected, structures are ranked
either by **hardness** or by how **contested** they are (how much the models disagree).
Hardness is the share of models for which the structure is an outlier: an error more
than three times that model's median error on the same benchmark, so it is unit-free
and rewards failures the models share. Only structures at least one model fails are
listed. With *n* models the share takes only *n* + 1 values, so many structures tie;
ties are broken by the structure's error percentile within its benchmark. Each
benchmark contributes at most two structures, so one benchmark cannot fill the list.
Only structures evaluated by at least three models are included.

Select a single model to switch the ranking to where *that* model's error is the largest
multiple of the median model's error on the same structure: the quickest way to find a
model's weak spots.
Click a bar to open the structure below; the *Open a structure from a list* disclosure
under the chart offers the same structures as buttons, so the 3D panel is reachable
from the keyboard too.


Selected structure
------------------

Whatever was last clicked in the head-to-head scatter or the failures chart is shown
here as an interactive 3D structure, with a table of every model's error on it.


Who fails alike
---------------

The heatmap correlates signed residuals (prediction minus reference) for each model
pair. Residuals are standardised per model and metric before pooling shared
observations. Metrics with more shared observations contribute more points. Hover for
the shared count. A cell is blank when there are fewer than ten comparable points
or the correlation is undefined.

Coverage rose
-------------

Each wedge shows the mean score for evaluated models on a benchmark. The area view
averages its benchmarks equally. Model coverage and scoring thresholds vary by
benchmark, and counts are shown on hover.


See also
--------

- :doc:`Benchmark scoring </developer_guide/scoring_and_normalisation>` — how raw metrics
  become the 0--1 scores the Explorer aggregates.
- :doc:`Levels of theory </developer_guide/levels_of_theory>` — the level-of-theory
  strings behind the in-domain/extrapolation labels.
- :doc:`Benchmarks </user_guide/benchmarks/index>` — what each benchmark measures.
