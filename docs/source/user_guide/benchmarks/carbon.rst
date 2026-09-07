======
Carbon
======

Lattice parameters
==================

Summary
-------

Performance in predicting lattice parameters, neighbour bond lengths, and energy
above graphite for eight carbon allotropes: graphite, graphene, diamond,
lonsdaleite, (9,0) and (9,9) carbon nanotubes, C60, and C100. Each model not
already trained with dispersion corrections is run twice, once with its plain
calculator and once with a D3 dispersion correction added; models trained on
dispersion run once, since D3 would be a no-op. The D3-corrected metrics carry the
benchmark score, the uncorrected metrics are reported alongside for comparison
(and are identical to the D3 metrics for dispersion-trained models).

Metrics
-------

1. Lattice parameter MAPE

For the six periodic systems, the DFT reference structure is relaxed (cell and
positions, ``fmax`` 1e-4 eV/Å for graphite, 1e-3 eV/Å otherwise). The relaxed and
reference lattice parameters are each divided by the system's supercell repeat factor
(graphite a/6 and c/2; diamond a/3; graphene a/5; lonsdaleite a/2 and c/2; NT(9,0)
c/5; NT(9,9) c/1) before the mean absolute percentage error is taken across all
resulting values. Reference cells were relaxed at ``ISIF = 2`` (ions only, fixed
cell) for every system except Lonsdaleite (``ISIF = 3``, cell and ions); this
benchmark relaxes the model's cell fully in every case, so the model is not
constrained the same way DFT was for those six systems.

2. Bond length MAPE

For C60 and C100, only atomic positions are relaxed (``fmax`` 1e-3 eV/Å). Interatomic
distances are binned into a first shell (< 1.6 Å) and a second shell (2.2-2.6 Å), and
each shell's mean (unrounded) bond length is compared between the relaxed structure
and the DFT reference structure.

3. Energy above graphite MAE

Reference INCARs for every system use ``ISPIN = 1`` (no spin polarisation) for the
isolated carbon atom, giving an atomic reference energy well above the spin-polarised
ground state; comparing to it directly scores every physically correct model as
wrong. Energy is therefore measured relative to graphite instead, so the
isolated-atom reference cancels:

``energy_above_graphite_ev_per_atom = E(X)/n_X - E(graphite)/n_graphite``

computed identically for model and reference (reference energies come from each
system's shipped ``REF_energy``). This is positive for every system less bound than
graphite (all seven here), following the standard "energy above hull" sign
convention. Graphite is 0 by construction and is excluded from the metric; the other
seven systems contribute. Reference values span 72-450 meV/atom, small enough that a
percentage error is dominated by whichever system happens to have the smallest
denominator, so this metric is reported as a mean absolute error in meV/atom rather
than as a MAPE.

Computational cost
------------------

Medium: relaxations of up to 288 atoms, run twice per model (plain and D3-corrected)
except for models already trained with dispersion corrections, which run once.

Data availability
-----------------

Input and reference structures:

* Rowe, P. et al. An accurate and transferable machine learning potential for carbon.
  *J. Chem. Phys.* **153**, 034702 (2020). https://doi.org/10.1063/5.0005084
* optB88-vdW exchange-correlation functional, PAW pseudopotentials, 500 eV plane-wave
  cutoff, 0.125 Å⁻¹ k-point spacing (VASP). ``ISIF = 2`` for every system except
  Lonsdaleite (``ISIF = 3``); ``ISPIN = 1`` throughout.
* Data repository: https://github.com/patrickwrowe/Carbon_GAP

Corrections to the original GAP-20 test suite:

* The (9,0) and (9,9) nanotube results were transposed in the original results
  dictionary; each nanotube now reports its own values.
* The original C100 bond-length comparison used the unrelaxed structure's distances
  instead of the relaxed structure's; the relaxed structure is now used, consistent
  with C60.
* The original script read a 4-atom unoptimised graphite cell
  (``Graphite_Unit_Unopt.POSCAR``) while still applying the a/6, c/2 divisors; the
  port uses the 288-atom ``Bulk_Structures/Graphite`` DFT-relaxed cell, for which
  those divisors are correct.
* The original (9,9) nanotube reference was a 174-atom defect-generation working
  copy with six two-coordinated (under-relaxed) sites, contracted by 2.8% on
  relaxation. It has been replaced with an intact 36-atom unit cell (all sites
  3-coordinated) at the same level of theory; its supercell divisor is 1, not 5.
* Energy above graphite is measured relative to graphite rather than to the
  original hardcoded per-system literals, which were built on the spin-unpolarised
  (``ISPIN = 1``) isolated-atom convention above and scored every correct model as
  wrong; see Metric 3.
* The original hand-written ``do_lonsdaleite`` scored ``a`` only. The port also
  scores ``c`` (divisor 2), since the shipped reference cell carries it and every
  other non-cubic system here reports ``c``; omitting it for lonsdaleite alone
  would be an inconsistency. Deliberate, one-line-reversible departure from the
  source.
