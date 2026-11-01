==========
Landscapes
==========

Benchmarks built on Landscape17, a set of potential energy conformational landscapes
for six rMD17 molecules computed at wB97X/6-31G(d). Each landscape is a network
of minima connected by transition states: 28 minima and 67 transition states in total,
with a DFT Hessian for every stationary point.

Results are kept per molecule rather than pooled. Aspirin alone is 37 of the 67
transition states and 6,944 of the 11,173 path frames, so a single pooled number
is largely a statement about aspirin.

Drift, overlap and RMSD are reported as positions in a distribution -- a median
and an upper percentile -- rather than as the fraction falling below a threshold.
A fraction is brittle in a way that misleads: a model that reproduces every
structure with the same small systematic displacement scores zero if that
displacement sits a hair outside the cut, which reads as total failure rather
than as the tidy, correctable offset it is. Percentiles say where the
distribution sits and whether it has a tail, and both can be given thresholds
with a physical justification.

Both benchmarks run the model in float64. That is a requirement rather than a
preference: eigenvector following takes its curvature from finite differences of
forces, which in float32 is noise-dominated, and the search wanders.

.. _stationary-points:

Stationary points
=================

Summary
-------

The model is handed the DFT stationary points and judged on what it does with
them. Nothing is searched for: every geometry is supplied.

Metrics
-------

1. Energies and forces along the steepest-descent paths

   The 11,173 frames joining the stationary points are ordinary points on the
   surface with non-zero forces, so this is a conventional accuracy test -- but
   sampled along the reaction coordinates rather than from a thermal
   distribution, which is where a landscape benchmark should want to look.

2. Residual force at each stationary point

   DFT forces are zero there by construction, so there is no reference to
   compare against. What the model reports is the residual: how far from
   stationary the DFT geometry is under this model. It predicts whether the
   structure will survive reconvergence, and it is not interchangeable with the
   drift below -- a model can disagree sharply along a stiff direction and
   barely move, or mildly along a soft one and travel far.

3. Drift on reconvergence

   Each reference minimum is relaxed under the model and each transition state
   refined by eigenvector following. A good model leaves the structure where it
   is; the distance it travels is the metric. Saddles are the harder case,
   being stationary in only one direction, and a model may have no saddle
   nearby at all.

4. Connectivity

   Whether a reconverged transition state still joins the same two minima.
   This is reported with the saddle reconvergence rather than as a test of its
   own, because eigenvector following already returns the minima its
   steepest-descent paths reach; separating them would mean running the whole
   search twice for a result already in hand.

5. Hessian comparison

   Normal mode overlaps against the DFT Hessian, the overlap of the reaction
   coordinate at saddles, and whether the model's curvature is systematically
   scaled.

.. _landscape-search:

Landscape search
================

Summary
-------

The harder question: given one conformer and no further reference
information, how much of the landscape does the model find on its own, and how
accurately?

A model can hold every stationary point it is handed and still fail to locate
them unaided, and it can also invent minima the reference does not contain.

This tests the model and the search together, which is the point -- a potential
is only useful for exploring a surface if a search can traverse it -- but it
means the search settings form part of the benchmark and are reported with the
results rather than left as defaults. All models use the same settings.

Metrics
-------

1. Minima and transition states recovered

   The fraction of reference stationary points the search locates, matched
   one-to-one by permutation-aware RMSD.

2. RMSD to the matched reference structure

   How accurately the recovered structures are placed.

3. Connectivity preserved

   Of the transition states recovered, the fraction joining the same pair of
   minima as in the reference.

4. Off-reference minima

   Structures the search converges to that match no reference minimum. These
   are not automatically wrong -- the reference is not guaranteed complete --
   but a model producing many of them is placing minima where DFT has none.
