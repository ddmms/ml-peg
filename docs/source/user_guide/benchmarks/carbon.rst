======
Carbon
======

Lattice parameters
==================

Summary
-------

Performance in predicting lattice parameters, neighbour bond lengths, and energy
above graphite for seven carbon allotropes: graphite, graphene, diamond,
lonsdaleite, the (9,0) carbon nanotube, C60, and C100. Each model not
already trained with dispersion corrections is run twice, once with its plain
calculator and once with a D3 dispersion correction added; models trained on
dispersion run once, since D3 would be a no-op. The D3-corrected metrics carry the
benchmark score, the uncorrected metrics are reported alongside for comparison
(and are identical to the D3 metrics for dispersion-trained models).

Metrics
-------

1. Lattice parameter MAPE

For the five periodic systems, the DFT reference structure is relaxed (cell and
positions, ``fmax`` 1e-4 eV/Å for graphite, 1e-3 eV/Å otherwise). The relaxed and
reference lattice parameters are each divided by the system's supercell repeat factor
(graphite a/6 and c/2; diamond a/3; graphene a/5; lonsdaleite a/2 and c/2; NT(9,0)
c/5) before the mean absolute percentage error is taken across all
resulting values. Reference cells were relaxed at ``ISIF = 2`` (ions only, fixed
cell) for every system except Lonsdaleite (``ISIF = 3``, cell and ions); this
benchmark relaxes the model's cell fully in every case, so the model is not
constrained the same way DFT was. Graphite, graphene and lonsdaleite sit within
0.25 GPa of their optB88-vdW minima; diamond's fixed cell carries -2.4 GPa, worth
about 0.2% in ``a``, and NT(9,0)'s axial repeat is an idealised 3 × 1.42 Å build.

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
graphite (all six here), following the standard "energy above hull" sign
convention. Graphite is 0 by construction and is excluded from the metric; the other
six systems contribute. Reference values span 72-450 meV/atom, small enough that a
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
  cutoff (VASP). Monkhorst-Pack k-point grids: graphite and diamond 2×2×2; graphene
  and lonsdaleite 3×3×3; NT(9,0) 1×1×2; C60 and C100 1×1×1.
  ``ISIF = 2`` for every system except Lonsdaleite (``ISIF = 3``); ``ISPIN = 1``
  throughout.
* Data repository: https://github.com/patrickwrowe/Carbon_GAP

Differences from the original GAP-20 test suite:

* The original C100 bond-length comparison used the unrelaxed structure's distances
  instead of the relaxed structure's; the relaxed structure is now used, consistent
  with C60.
* The original script read a 4-atom unoptimised graphite cell
  (``Graphite_Unit_Unopt.POSCAR``) while still applying the a/6, c/2 divisors; the
  port uses the 288-atom ``Bulk_Structures/Graphite`` DFT-relaxed cell, for which
  those divisors are correct.
* The (9,9) nanotube is not included. Its reference cell is a 174-atom
  defect-generation working copy carrying six two-coordinated sites, and the intact
  cell available at the same level of theory is an idealised build with neither its
  positions nor its cell relaxed. Neither yields a lattice parameter. The tube is
  covered by the nanotube formation energies benchmark, where model and reference
  share that geometry.
* Energy above graphite is measured relative to graphite rather than to the
  original hardcoded per-system literals, which were built on the spin-unpolarised
  (``ISPIN = 1``) isolated-atom convention above and scored every correct model as
  wrong; see Metric 3.
* The original hand-written ``do_lonsdaleite`` scored ``a`` only. The port also
  scores ``c`` (divisor 2), since the shipped reference cell carries it and every
  other non-cubic system here reports ``c``; omitting it for lonsdaleite alone
  would be an inconsistency. Deliberate, one-line-reversible departure from the
  source.

Surface energies
================

Summary
-------

Performance in predicting as-cut and relaxed surface energies for three carbon
surfaces: diamond {100}, graphite (0001), and amorphous carbon. Every value is a
single-point evaluation of the model's calculator at a shipped DFT reference
geometry; the benchmark performs no geometry optimisation. Each model not already
trained with dispersion corrections is run twice, once with its plain calculator and
once with a D3 dispersion correction added; models trained on dispersion run once,
since D3 would be a no-op. The D3-corrected metrics carry the benchmark score, the
uncorrected metrics are reported alongside for comparison (and are identical to the
D3 metrics for dispersion-trained models).

Graphite (0001) is a cleave between basal planes held together only by dispersion.
Without a dispersion correction its surface energy is close to zero; the
uncorrected column is expected to look poor for that surface specifically.

Metrics
-------

1. As-cut surface energy MAE

For diamond {100} and graphite (0001), the shipped as-cut and relaxed reference
geometries each add 15 Å of vacuum along the cell's third vector to the shipped bulk
cell (equivalent to the cell's z-component for these references, since the two
coincide here); the model evaluates a single point at each of the three shipped
geometries, unmodified:

``surface_energy_j_m2 = 0.5 * (E_slab - E_bulk) / |a1 x a2| * 16.0218``

where ``a1`` and ``a2`` are the first two vectors of the shipped reference bulk
cell, and 16.0218 converts eV/Å² to J/m².

Amorphous carbon has no unique cleavage plane, so its reference is an ensemble of
ten independently generated bulk configurations, each cut along four or five
different planes (49 cuts total). Every (bulk, slab) pair uses the formula above at
the shipped, unrelaxed atomic positions; the as-cut surface energy is the mean over
all 49 pairs. This port reports every surface energy, including the amorphous
ensemble average, as a positive number via ``0.5 * (E_slab - E_bulk) / area``; the
2021 amorphous script applied the same subtraction with the operands reversed and
plotted the absolute value of both quantities, which is a display convention that
cancels exactly in the relative error it reported and changes no result.

The reported error is the mean absolute error across all three surfaces:

``As-cut surface energy MAE = mean(|ref_surface_energy_j_m2 - surface_energy_j_m2|)``

The original test suite reports the signed fractional error
``(ref - model) / ref``; this benchmark uses the absolute error in J/m² instead (see
Metric 2 in the lattice parameters section above for why a relative error is not
used here), so the sign convention does not carry over.

2. Relaxed surface energy MAE

For diamond {100} and graphite (0001), the model evaluates a single point at the
shipped relaxed reference geometry. The two provenances differ. Every diamond INCAR
under ``Surfaces/DFT_Reference/Diamond/`` has ``NSW = 0`` and takes its geometry
from a GAP-20 POSCAR, so DFT never performed its own ionic relaxation for diamond;
its "relaxed" reference is a single-point at a distinct GAP-20 geometry, evaluated
the same way as its as-cut reference. Graphite's relaxed reference is a genuine DFT
ionic relaxation:
``Graphite/0001/actual_relaxed/INCAR`` has ``NSW = 1000``, and its geometry moves up
to 0.104 Å (mean 0.050 Å over 48 of 60 atoms) between the as-cut and relaxed frames.
Either way, this benchmark takes no relaxation action itself — only a single point
at whichever geometry is shipped. Amorphous carbon is excluded from this metric: its
DFT reference used only unrelaxed cuts (``surfaces_unrelaxed``), so no relaxed
reference geometry exists for it at all.

Computational cost
------------------

Small: single-point evaluations only, no geometry optimisation. Three per model
variant for each of diamond {100} and graphite (0001) (bulk, as-cut, relaxed) — six
combined — plus up to 59 for the amorphous ensemble (10 bulk, 49 slab, up to 216
atoms each): 65 per variant, 130 total for models run both plain and D3-corrected.

Data availability
-----------------

Input and reference structures:

* Rowe, P. et al. An accurate and transferable machine learning potential for carbon.
  *J. Chem. Phys.* **153**, 034702 (2020). https://doi.org/10.1063/5.0005084
* optB88-vdW exchange-correlation functional, PAW pseudopotentials, 500 eV plane-wave
  cutoff (VASP). Monkhorst-Pack k-point grids: diamond {100} 15×15×2 (bulk),
  15×15×1 (as-cut and relaxed); graphite (0001) 7×7×1 throughout; amorphous carbon
  2×2×2 (bulk), 2×2×1 (slab).
* Data repository: https://github.com/patrickwrowe/Carbon_GAP

Differences from the original GAP-20 test suite:

* Diamond {111} is omitted. Its as-cut and relaxed DFT reference calculations were
  truncated mid-SCF (no closing ``</calculation>`` or ``</modeling>`` tag, no final
  structure, no forces); the scheduler log shows the VASP job never ran. Only the
  bulk survived, which alone cannot give a surface energy. This exclusion is
  permanent: ``Diamond_Reconstructed`` exists in the source data but has different
  physics and no relaxed counterpart, and is not substituted in.
* Diamond {110} is also omitted. Its reference slab is not a clean cleave:
  expanding the bulk cell along c splits a (110) layer that straddles the cell
  boundary, leaving a coordination-1 adatom on each face.

Nanotube formation energies
===========================

Summary
-------

Performance in predicting strain energy relative to graphene for ten armchair
(n, n) and ten zigzag (n, 0) carbon nanotubes, n = 5..14. Every value is a
single-point evaluation of the model's calculator at a shipped DFT reference
geometry; the benchmark performs no geometry optimisation. Each model not
already trained with dispersion corrections is run twice, once with its plain
calculator and once with a D3 dispersion correction added; models trained on
dispersion run once, since D3 would be a no-op. The D3-corrected metrics carry
the benchmark score, the uncorrected metrics are reported alongside for
comparison (and are identical to the D3 metrics for dispersion-trained
models).

Metrics
-------

1. Armchair strain energy MAE
2. Zigzag strain energy MAE

For each tube, the model and the reference are each a single-point evaluation
at the shipped reference geometry:

``strain_energy_ev_per_atom = E_tube / n_tube - E_graphene / n_graphene``

Each side uses its own graphene energy — the model's own graphene single-point
energy for the model series, the DFT ``REF_energy`` for the reference series —
so that the isolated-atom term cancels within each side rather than being
carried across the comparison. Graphene is a fixed reference value, not a
benchmarked system, and is excluded from both metrics and from the parity plot.

Reference strain energies span roughly 20-530 meV/atom, small enough that a
percentage error would be dominated by the thinnest, most strained tubes, so
the reported error is a mean absolute error in meV/atom rather than a MAPE,
split by chirality so a model weak on one series stays visible instead of
being averaged away:

``<Chirality> strain energy MAE = mean(|ref_strain_energy_ev_per_atom -
strain_energy_ev_per_atom|)``

mace-mp-0b3 underestimates strain energy across all 20 tubes, more so at
small diameter. This is model behaviour, not a benchmark artefact: the
model's own strain energy still scales as 1/d² (strain × d² is
near-constant at 5.20-5.45 across both families), fitting strain = A/d² + B
to each family separately gives intercept differences of opposite sign
between families (+3.1 armchair, -8.9 zigzag) which a constant pipeline
offset could not produce, and two other models run through the same path
do not share the direction of the discrepancy (mace-mpa-0 straddles zero,
mace-omat-0 overestimates by 25-46%).

Computational cost
------------------

Small: single-point evaluations only, no geometry optimisation. 20 tubes plus
one graphene reference, 21 single points per model variant, run twice per
model (plain and D3-corrected) except for models already trained with
dispersion corrections, which run once.

Data availability
-----------------

Input and reference structures:

* Rowe, P. et al. An accurate and transferable machine learning potential for carbon.
  *J. Chem. Phys.* **153**, 034702 (2020). https://doi.org/10.1063/5.0005084
* optB88-vdW exchange-correlation functional, PAW pseudopotentials, 500 eV plane-wave
  cutoff (VASP). Monkhorst-Pack k-point grid: 1×1×8 for all 20 nanotubes, 3×3×3 for
  the graphene reference. Every nanotube reference INCAR uses ``NSW = 1`` (single
  point).
* Data repository: https://github.com/patrickwrowe/Carbon_GAP

Differences from the original GAP-20 test suite:

* The original ``nanotubes_formation_energy/test.py`` scores each tube
  against a hardcoded isolated-atom reference: the spin-unpolarised DFT
  carbon atom (``single_atom_energy = 0.94664775`` eV) for the reference
  series, and the model's own isolated-atom energy, computed at runtime by
  ``get_model_single_atom_energy()``, for the model series. That convention
  was self-consistent for GAP-20, the model it was built around, but does
  not generalise: an arbitrary model's own isolated-atom energy carries
  whatever spin convention that model was trained on, not the reference's
  spin-unpolarised one. This benchmark instead reports strain energy
  relative to graphene, with each side using its own graphene energy, so
  the isolated-atom convention never enters either series; see Metrics
  above. The published archive also no longer ships an isolated atom to
  compute either side from.
