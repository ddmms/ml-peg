======
Carbon
======

Structural and energetic properties of carbon allotropes, ported from the test suite
used to validate GAP-20. All three benchmarks share one reference dataset:

* Rowe, P. et al. An accurate and transferable machine learning potential for carbon.
  *J. Chem. Phys.* **153**, 034702 (2020). https://doi.org/10.1063/5.0005084
* optB88-vdW exchange-correlation functional, PAW pseudopotentials, 500 eV plane-wave
  cutoff (VASP), ``ISPIN = 1`` throughout.
* Data repository: https://github.com/patrickwrowe/Carbon_GAP

Each model not already trained with dispersion corrections is run twice, plain and
with a D3 correction added; models trained on dispersion run once, since D3 would be
a no-op. The D3-corrected metrics carry the benchmark score; the uncorrected metrics
are reported alongside at zero weight.

Lattice parameters
==================

Summary
-------

Lattice parameters, neighbour bond lengths, and energy above graphite for seven
allotropes: graphite, graphene, diamond, lonsdaleite, the (9,0) nanotube, C60 and
C100.

Metrics
-------

1. Lattice parameter MAPE

The five periodic systems are relaxed, cell and positions (``fmax`` 1e-4 eV/Å for
graphite, 1e-3 eV/Å otherwise). Relaxed and reference parameters are each divided by
the system's supercell repeat factor (graphite a/6 and c/2; diamond a/3; graphene
a/5; lonsdaleite a/2 and c/2; NT(9,0) c/5) before the mean absolute percentage error
is taken across all resulting values.

Reference cells were relaxed at fixed cell (``ISIF = 2``) except lonsdaleite, so the
model is less constrained than DFT was. Graphite, graphene and lonsdaleite sit within
0.25 GPa of their optB88-vdW minima; diamond's cell carries -2.4 GPa, about 0.2% in
``a``, and NT(9,0)'s axial repeat is an idealised 3 × 1.42 Å build. These set the
resolution floor of the metric.

2. Bond length MAPE

For C60 and C100 only atomic positions are relaxed (``fmax`` 1e-3 eV/Å). Interatomic
distances are binned into a first shell (< 1.6 Å) and a second shell (2.2-2.6 Å), and
each shell's mean bond length is compared to the DFT reference.

3. Energy above graphite MAE

Energy is measured relative to graphite, the stable allotrope:

``energy_above_graphite_ev_per_atom = E(X)/n_X - E(graphite)/n_graphite``

computed identically for model and reference. Graphite is 0 by construction and
excluded; the other six systems contribute. Reference values span 72-450 meV/atom, so
a percentage error would be dominated by the smallest denominator and this is
reported as a mean absolute error in meV/atom.

Computational cost
------------------

Medium: relaxations of up to 288 atoms, per model variant.

Data availability
-----------------

Monkhorst-Pack k-point grids: graphite and diamond 2×2×2; graphene and lonsdaleite
3×3×3; NT(9,0) 1×1×2; C60 and C100 1×1×1.

Surface energies
================

Summary
-------

As-cut and relaxed surface energies for diamond {100}, graphite (0001) and amorphous
carbon. Every value is a single-point evaluation of the model's calculator at a
shipped DFT reference geometry; the benchmark performs no geometry optimisation.

Graphite (0001) is a cleave between basal planes held together only by dispersion, so
its uncorrected column is expected to look poor for that surface specifically.

Metrics
-------

1. As-cut surface energy MAE

For diamond {100} and graphite (0001), the model evaluates a single point at each of
the three shipped geometries — bulk, as-cut and relaxed — unmodified:

``surface_energy_j_m2 = 0.5 * (E_slab - E_bulk) / |a1 x a2| * 16.0218``

where ``a1`` and ``a2`` are the first two vectors of the reference bulk cell, and
16.0218 converts eV/Å² to J/m².

Amorphous carbon has no unique cleavage plane, so its reference is ten independently
generated bulk configurations, each cut along four or five planes (49 cuts); its
as-cut value is the mean over all 49 (bulk, slab) pairs.

The reported error is the mean absolute error across all three surfaces.

2. Relaxed surface energy MAE

A single point at the shipped relaxed geometry for diamond {100} and graphite (0001).
The provenances differ: every diamond reference INCAR has ``NSW = 0``, so its
"relaxed" reference is a single point at a distinct GAP-20 geometry, whereas
graphite's is a genuine DFT ionic relaxation (``NSW = 1000``, atoms moving up to
0.104 Å). The benchmark relaxes nothing either way. Amorphous carbon is excluded: its
DFT reference used only unrelaxed cuts, so no relaxed reference exists.

Computational cost
------------------

Small: 65 single points per model variant — three each for diamond {100} and graphite
(0001), plus 59 for the amorphous ensemble (10 bulk, 49 slab, up to 216 atoms each).

Data availability
-----------------

Monkhorst-Pack k-point grids: diamond {100} 15×15×2 (bulk) and 15×15×1 (slabs);
graphite (0001) 7×7×1 throughout; amorphous carbon 2×2×2 (bulk) and 2×2×1 (slab).
Surface references add 15 Å of vacuum along the third cell vector.

Nanotube formation energies
===========================

Summary
-------

Strain energy relative to graphene for ten armchair (n, n) and ten zigzag (n, 0)
carbon nanotubes, n = 5..14. Every value is a single-point evaluation of the model's
calculator at a shipped DFT reference geometry; the benchmark performs no geometry
optimisation.

Metrics
-------

1. Armchair strain energy MAE
2. Zigzag strain energy MAE

``strain_energy_ev_per_atom = E_tube / n_tube - E_graphene / n_graphene``

Each side uses its own graphene energy: the model's own graphene single point for the
model series, the DFT ``REF_energy`` for the reference series. Graphene is a fixed
reference value, not a benchmarked system, and is excluded from both metrics and from
the parity plot.

Reference strain energies span roughly 20-530 meV/atom, so the reported error is a
mean absolute error in meV/atom for the reason given under lattice parameters Metric
3, split by chirality so a model weak on one series is not averaged away.

Computational cost
------------------

Small: 21 single points per model variant, 20 tubes plus one graphene reference.

Data availability
-----------------

Monkhorst-Pack k-point grids: 1×1×8 for all 20 nanotubes, 3×3×3 for the graphene
reference. Every nanotube reference INCAR uses ``NSW = 1`` (single point).
