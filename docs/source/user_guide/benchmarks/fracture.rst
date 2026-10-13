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

Each model traces its own S-curve by pseudo-arclength continuation in (u, K)
(``matscipy.fracture_mechanics.continuation.trace_s_curve``), using only the model's
forces: no model Hessian is needed, as the Newton corrector uses a Stillinger-Weber
Hessian with Broyden updates, falling back to Newton-Krylov. The continuation runs on
clamped K-field clusters with the topology of the DFT reference
(``clusters.cluster_like``): every model sees the same atoms, H caps and tip bond, with
its own lattice constant, cubic elastic constants and surface energy (computed in the
benchmark), so K is measured in units of the model's own Griffith value K_G. Three crack
systems are included: (110)[1-10], (110)[001] and (111)[1-10].

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
coverage times the score of its error on the covered part. Thresholds changed in the
app do not change this coverage padding, which uses the defaults. A calculation that
fails (an error in the constants, the cleavage energy, the tracer or the scoring of a
case) is not scored: its metrics and benchmark score are NaN. The benchmark score (the
mean of the nine metric scores with default weights) is the combined value.

Thresholds
----------

The thresholds are anchored on the Stillinger-Weber (SW) potential (F. H. Stillinger and
T. A. Weber, Phys. Rev. B 31, 5262 (1985)), a long-established empirical model of
silicon with a physically understood failure for fracture: it reproduces the elastic
constants but not the bond-breaking at the crack tip, and is known to give qualitatively
wrong crack behaviour. Unlike recent MLIPs, it is a fixed baseline that does not change
between model releases. For each error term, a tolerance tau is chosen so that the
median SW error against DFT over the three crack systems is 3 tau, and the thresholds
are good = tau and bad = 3 tau. A model as poor as SW therefore scores about 0, and a
model whose errors are within a third of SW's scores 1. The values are tau = 0.0803
(K, in K/K_G), 0.0952 (E, in 2 gamma L b_bulk) and 0.144 (D). They are not derived from
an estimate of the DFT uncertainty.

Benchmark speed
---------------

Fast: about 7 minutes for ``mace-mh-1-omat`` on an RTX A4500 GPU (constants in seconds,
then 2-3 minutes per crack system, 150-200 continuation points at 3-4 force calls each).

Data availability
-----------------

Reference data:

* Spin-polarised PBE (VASP, PAW, 520 eV cutoff) pseudo-arclength continuation on
  H-passivated clamped clusters (100-135 atoms), packaged as the reference paths of the
  three cases.
* Manuscript in preparation.

Method:

* M. Buze and J. R. Kermode, Numerical-continuation-enhanced flexible boundary
  condition scheme applied to mode-I and mode-III fracture, Phys. Rev. E 103, 033002
  (2021).
* J. R. Kermode et al., Low-speed fracture instabilities in a brittle crystal,
  Nature 455, 1224 (2008).
* T. D. Swinburne and J. R. Kermode, Computing energy barriers for rare events from
  hybrid quantum/classical simulations through the virtual work principle,
  Phys. Rev. B 96, 144102 (2017).

Tool and computational resource disclosure
------------------------------------------

The benchmark code, its tests and this page were written with an AI coding assistant
(Claude, Anthropic). The implementer manually inspected and verified the code and the
tests and takes responsibility for them. The DFT reference calculations
used the Avon (University of Warwick) and Sulis (HPC Midlands+) clusters.
