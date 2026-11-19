==========
Landscapes
==========

Benchmarks built on Landscape17, a set of potential energy conformational landscapes
for six rMD17 molecules computed at ωB97X/6-31G(d). Each landscape is a network
of minima connected by transition states: 28 minima and 67 transition states in total,
with a DFT Hessian for every stationary point.

Every number in the tables is pooled over the six molecules, and so leans
heavily on aspirin: aspirin alone is 37 of the 67 transition states and 6,944
of the 11,173 path frames.

Every benchmark here runs the model in float64. That is a requirement rather than a
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

2. Energy and force at each stationary point

   The energy error at the 95 DFT minima and transition states, each method
   measured from its own per-molecule mean so that a constant offset cancels.

   DFT forces are zero there by construction, so there is no reference to
   compare against. What the model reports is the residual: how far from
   stationary the DFT geometry is under this model, as an RMSE over every
   force component, pooled the same way as along the paths. It predicts
   whether the structure will survive reconvergence, and it is not
   interchangeable with the RMSD on reconvergence below -- a model can
   disagree sharply along a stiff direction and barely move, or mildly along
   a soft one and travel far.

3. RMSD on reconvergence

   Each reference minimum is relaxed under the model and each transition state
   refined by eigenvector following. A good model leaves the structure where it
   is; the distance it travels is the metric. Saddles are the harder case,
   being stationary in only one direction, and a model may have no saddle
   nearby at all. Saddles converged is the fraction of the 67 for which
   eigenvector following reaches a saddle within its step budget, and the
   saddle distances are taken over those alone.

4. Connectivity preserved

   The fraction of all 67 reference transition states whose reconverged saddle
   still joins the same two minima as in DFT, so a saddle that fails to
   reconverge counts against it. This is reported with the saddle
   reconvergence rather than as a test of its own, because eigenvector
   following already returns the minima its steepest-descent paths reach;
   separating them would mean running the whole search twice for a result
   already in hand.

5. Hessian comparison

   Normal mode overlaps against the DFT Hessian, the overlap of the reaction
   mode at saddles, and how much softer or stiffer than DFT the model is
   overall, in percent. It also reports the curvature error over the softest
   third of the modes, where a landscape lives, and the ratio of the model's
   imaginary eigenvalue to DFT's, which sets the tunnelling correction and the
   transition state theory prefactor. That ratio is scored as
   ``min(r, 1/r)``, so a barrier top too sharp counts against a model as much
   as one too flat by the same factor.
