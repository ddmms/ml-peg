========
Fracture
========

Brittle crack
=============

Summary
-------

Performance in reproducing the lattice-trapping S-curves of crack tips in silicon: the
stress intensity factor K along the full crack path (stable, metastable and unstable
states), the bond-breaking energy and the displacement field near the tip, compared
with spin-polarised DFT (PBE) on the same clamped crack clusters.

Each model traces its own S-curve by force-only pseudo-arclength continuation
(``matscipy.fracture_mechanics.continuation.trace_s_curve``) on clamped K-field
clusters with the topology of the DFT reference (``clusters.cluster_like``): every model
sees the same atoms, H caps and tip bond, with its own lattice constant, cubic elastic
constants and surface energy (computed in the benchmark), so K is measured in units of
the model's own Griffith value K_G. Three crack systems are included:
(110)[1-10], (110)[001] and (111)[1-10].

Metrics
-------

Each case contributes three metrics. All are compared on the DFT reference path, as a
function of a bond length near the tip (the case's reaction coordinate) in units of the
model's own bulk bond, over the whole DFT range (the scored window, the same for every
model).

1. K error

   RMS difference of K/K_G between model and DFT along the window.

2. E error

   RMS difference of the bond-breaking energy along the path, in units of
   2 gamma L b_bulk (gamma: cleavage energy, L: cluster length along the crack front,
   b_bulk: bulk bond length). The elastic energy c k^2 (k = K/K_G) is removed with one
   coefficient for all models, fitted on the DFT reference's closed branch; in these
   units it depends only on the cluster geometry for a linear-elastic material.

3. D error

   RMS energy-norm error of the crack-tip displacement field: the strain of the
   displacements relative to each model's own K-field, on second-neighbour bonds within
   8 Å of the tip and not behind it, matched at equal coordinate.

A model that does not cover the whole window (its trace stops early or folds back) is
penalised: the uncovered fraction counts at the "bad" threshold, so its score is the
coverage times the score of its error on the covered part. The thresholds are the
calibrated tolerance (good) and three times it (bad); Stillinger-Weber, the clearly poor
reference model, has a median error of three tolerances. Thresholds changed in the app
do not change this coverage padding, which uses the defaults. The benchmark score (the
mean of the nine metric scores with default weights) is the combined value.

Benchmark speed
---------------

Slow: about 7 minutes for ``mace-mh-1-omat`` on an RTX A4500 GPU (constants in seconds,
then 2-3 minutes per crack system, 150-200 continuation points at 3-4 force calls each).

Data availability
-----------------

Reference data:

* Spin-polarised PBE (VASP, PAW, 520 eV cutoff) pseudo-arclength continuation on
  H-passivated clamped clusters (100-135 atoms), packaged as the reference paths of the
  three cases.
* Manuscript in preparation.
