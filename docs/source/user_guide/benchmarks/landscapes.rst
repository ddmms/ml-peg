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
   mode at saddles, and whether the model's curvature is systematically
   scaled. It also reports the curvature error over the softest third of the
   modes, where a landscape lives, and the ratio of the model's imaginary
   eigenvalue to DFT's, which sets the tunnelling correction and the
   transition state theory prefactor.

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

   How accurately the recovered structures are placed, over the matched pairs
   only and so always within the 0.3 Å matching radius.

3. Connectivity preserved

   The fraction of all 67 reference transition states that the search both
   recovered and found joining the same pair of minima as in the reference.
   Taken over every reference rather than the recovered ones, so a model that
   finds few cannot score well on the few it finds.

4. Off-reference minima

   Minima whose bonding differs from the molecule's. The search only sets out
   to change conformation, with basin-hopping steps that rotate dihedral
   angles, so in an ideal case the bonding framework never changes. These are
   not automatically wrong -- a model may find a genuine isomer and only place
   it at the wrong energy -- but a model producing many of them lets bonds
   break or form where only the conformation should change. A correctly bonded
   minimum that matches no reference minimum is unmatched, not off-reference.

5. Distorted minima

   Minima holding one of the molecule's own bonds more than 20% away from its
   length in the reference. The off-reference test compares bonding graphs
   and cannot see these: a bond crushed to two thirds of its length is still
   a bond, so the structure reads as the right molecule at an impossible
   geometry. The flagged structures are written out beside the model's
   results.

6. RMSD to the nearest found structure

   For every reference minimum and transition state, the distance to the
   closest structure the model found, matched or not. Unlike item 2 this does
   not condition on success, so it does not flatter a model that found two
   structures precisely and missed the rest.

7. Minima near-missed

   Reference minima whose closest found structure lies between the 0.3 Å
   matching radius and 0.45 Å. Recovery is a pass rate over a hard radius, so
   a reference missed by 0.31 Å scores the same as one missed by 3 Å; this
   separates the two. Low recovery with many near misses means the model is
   systematically displaced rather than failing to find the structures, which
   is a correctable offset.
