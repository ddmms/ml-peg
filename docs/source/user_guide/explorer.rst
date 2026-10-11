==============
Model Explorer
==============

The category tabs answer "how well does each model do on *this* benchmark?". The
Model Explorer, reached from the ``Explorer`` link in the sidebar (``/explorer``), reads
across all of them to answer the two questions a user actually has:

- Which model should I use for my system?
- Where will it stop working?

Everything on the page is derived at app start-up from the benchmark data already
shipped with the app (the parity-plot figures, metrics tables and ``.xyz`` geometries in
``ml_peg/app/data``). There is no separate build step: a newly added benchmark appears in
the Explorer automatically.

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
   * - vs CCSD(T)/MP2
     - The reference is a higher-level method (coupled cluster, DMC, ...) than the
       DFT the model was trained on.
   * - vs experiment
     - The reference is experimental.
   * - soundness
     - No reference level applies (physicality checks).

A benchmark with several metrics takes the most severe mismatch among them. Anything
other than *in-domain* is an extrapolation. This is not a verdict on the model, but it
tells you how much the score should be trusted to transfer to your own setting.


Model finder
------------

Describe what you are simulating and get a ranked shortlist of models with the evidence
behind each pick.

Controls
^^^^^^^^

- **What are you simulating?** — an experiment preset that selects a bundle of
  benchmark areas: *Anything*, *Bulk materials & crystals*, *Surfaces, defects &
  reactions*, *Molecules & non-covalent chemistry*, or *Metals & f-block chemistry*.
- **Elements in your system** — one or more element symbols. When set, the finder adds
  evidence from every test structure containing those elements.
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
2. **Structure evidence** (only when elements are selected): on every test structure
   containing those elements, the models run on it are ranked by error, and the model's
   average standing is its *mean win rate* (after HELM, Liang *et al.* 2022): the fraction
   of other models it beats. It is averaged within each benchmark first, so a benchmark
   with thousands of structures counts the same as a small one. This depends on which
   models were compared, so it will shift as models are added.

The fit is then

.. math::

   \text{fit} = 0.6 \times \text{benchmark} + 0.4 \times \text{structure}

or just the benchmark evidence when no elements are chosen. Two safeguards stop thin
evidence from winning:

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

Compare two models structure by structure over everything they were both evaluated on.
A model "wins" a structure when its absolute error is smaller. Since both models are
measured on the same structure, this is a fair, unit-free comparison (the sign test of
Demšar 2006).

The chart shows one horizontal bar per category:

- **Direction** points to the model that was closer to the reference more often.
- **Length** is the margin: how far the win rate is from 50/50.
- **Whiskers** are 95% Wilson intervals, so a bar backed by a few structures reads as
  less certain than one backed by thousands. They assume structures are independent,
  which they are not within a benchmark, so they are somewhat optimistic.
- **Grey bars** fall inside the ±5% practical-tie band and are too close to call.

Win rates are averaged with *equal weight per benchmark* rather than pooled over raw
structures, so one benchmark with thousands of structures cannot swamp the verdict for
its category. Exact ties are shown in the counts but excluded from every win rate, so
the two models' rates always sum to one.

The win rate counts wins, not how large each win was. Click a bar, or choose an area in
the *Open an area's structures* picker below the chart, to open that category's
structure-by-structure scatter, where each point is one structure and the axes are each
model's error as a multiple of that benchmark's median structure error (so benchmarks
with different units share an axis). Click any point to load the structure into the
*Selected structure* panel.


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

A model-by-model heatmap of error correlation. Each model's per-structure errors are
standardised within every benchmark and correlated against every other model's over the
structures they share. Warm cells are pairs that make the same mistakes on the same
structures; cool cells are complementary. Rows and columns are ordered by hierarchical
clustering, so models from the same family or training set appear as blocks. If two
models fail alike, combining them in an ensemble will add little.


Coverage rose
-------------

A polar overview of the field rather than of any one model. Each wedge is a benchmark
area (or a single benchmark, using the *Group by* control) and its length and colour give
the mean score across every model evaluated there. Long green wedges are areas current
models already handle; short red wedges are open problems. Because each wedge averages
over whichever models were run in that area, read it as a rough guide rather than an
exact cross-area comparison.


See also
--------

- :doc:`Benchmark scoring </developer_guide/scoring_and_normalisation>` — how raw metrics
  become the 0--1 scores the Explorer aggregates.
- :doc:`Levels of theory </developer_guide/levels_of_theory>` — the level-of-theory
  strings behind the in-domain/extrapolation labels.
- :doc:`Benchmarks </user_guide/benchmarks/index>` — what each benchmark measures.
