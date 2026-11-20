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

   The model's Hessian is taken by central finite differences of its forces,
   with a 1e-3 Å step, at the DFT geometry. Both Hessians are compared with
   translations and rotations projected out, and modes are matched one to one
   by overlap.

Scoring
-------

Path energy RMSE, Path force RMSE and the stationary-point Force RMSE are
reported but not scored. The path errors track the stationary-point ones almost
exactly, so scoring both would count the same thing twice. Force RMSE mostly
measures the gap between levels of theory rather than the model: the reference
geometries are not stationary at other levels, and ωB97M-V/def2-TZVPD, the
OMol25 level of theory, itself has an RMS force of 129 meV/Å at them.

The thresholds come from the reference wherever it supplies one:

* **Energies.** *Bad* is what a model that knows nothing scores -- the same
  energy for every structure of a molecule -- which is the spread of the
  reference's own relative energies: 13.8 meV/atom at the stationary points and
  11.6 along the paths. *Good* is the reference uncertainty: scored against this
  reference, ωB97M-V/def2-TZVPD gets 1.98 meV/atom at the stationary points and
  1.51 on a 300-frame sample of the paths, so the thresholds are 2.0 and 1.5.
* **Forces** (unscored, so the thresholds only colour the columns). *Good* is
  the same measured gap, 129 meV/Å at the stationary points and 123 along the
  paths. *Bad* is 153 meV/Å, what predicting zero force along the paths scores.
* **Drift** (minima and saddle RMSD, median and 90th percentile). *Bad* is
  0.34 Å, the separation of the closest two distinct reference minima: past it,
  a structure that drifted cannot be told from one that landed on another
  minimum. *Good* is 0.01 Å, about the run-to-run noise of the saddle search on
  GPU, which leaves room for small geometry differences between functionals.
* **Coverage** (saddles converged, connectivity preserved). A perfect model of
  the reference scores 1 by construction, since every search starts at its own
  saddle. *Good* is 0.95 and 0.9, a judgement call leaving room for a model
  trained on another functional, whose surface may have lost or moved a few of
  these saddles. *Bad* is 0, what a model that finds nothing scores.
* **Mode overlaps.** 0.99 and 0.8, an 8° and a 37° misalignment, well above the
  0.32 that randomly oriented modes score.
* **Stiffness.** 2% and 20%, about 1% and 10% in frequency; 10% is the error of
  uncorrected Hartree-Fock frequencies.
* **Soft mode curvature.** 0.02 and 0.2 eV/Å², about 1% and 7% of the median
  curvature of the softest third of the reference modes, 2.7 eV/Å².
* **Reaction mode curvature.** *Good* is 1. *Bad* is 0.44, what a single
  imaginary eigenvalue predicted for every saddle scores, so a model at or
  below it shows no skill there.

The ωB97M-V/def2-TZVPD figures are energies and analytic forces at the
reference geometries (gpu4pyscf, direct SCF, (99, 590) grid, (50, 194) VV10
grid), scored exactly as a model is.

Protocol
--------

The settings are fixed in ``calc_stationary_points.py``:

* Minima are relaxed with L-BFGS until the largest gradient component falls
  below 1e-6 eV/Å, within 1000 iterations. A relaxation that stops on the
  iteration limit has not arrived, and contributes NaN.
* Transition states are refined by hybrid eigenvector following (topsearch)
  within 150 steps, with a gradient criterion of 1e-2 eV/Å. Each endpoint of
  its steepest-descent paths is identified with the nearest reference minimum
  within 0.3 Å. A saddle not reached within the budget counts against Saddles
  converged and Connectivity preserved; some reference saddles never converge
  even given 3000 steps.
* Every structure is a neutral singlet.

A model error on a structure is warned about and recorded rather than stopping
the run: it voids that model's metrics that depend on the structure, or, for a
saddle, counts as one not reached.

Benchmark speed
---------------

Slow: about 2.5 to 5 hours per model on one NVIDIA H200 for the MACE models
tested, almost all of it the saddle reconvergence (mace-mp-0a: 4.6 hours for the
saddles, half an hour for everything else).

Data availability
-----------------

Input structures:

* The 28 minima, 67 transition states and 11,173 steepest-descent path frames
  of Landscape17: Cărare, V. et al. Global properties of the energy landscape:
  a testing and training arena for machine learned potentials. npj Comput.
  Mater. (2025). https://doi.org/10.1038/s41524-025-01878-x

Reference data:

* Energies at the stationary points, and energies and forces along the paths:
  Landscape17, ωB97X/6-31G(d) in Psi4.
* Hessians at the 95 stationary points: computed for this benchmark in PySCF at
  the same level -- ωB97X/6-31G(d) with Cartesian d functions, restricted
  Kohn-Sham, (100, 770) grid, SCF converged to 1e-11. They are analytic,
  agree with finite differences (0.005 Å step) to within 0.4% on ethanol and
  malonaldehyde, and reproduce each Landscape17 energy to within 0.05 kJ/mol.
