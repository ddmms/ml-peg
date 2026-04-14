===========
Physicality
===========

Locality
========

Summary
-------

Performance in respecting locality, by measuring deviations from a neglible interactions
between acetone and distance atoms.

Metrics
-------

1. Maximum difference in force due to "ghost atoms"

Forces on an isolated acetone molecule are calculated, and the forces on the same atoms
are calculated after 20 Ne atoms are placed in a 60 Å cubic box, at least 40 Å from the
acetone's centre of mass. The magnitude of the maximum difference in force is reported.

2. Mean difference in force due to a distance hydrogen

Forces on an isolated acetone molecule are calculated, and the forces on the same atoms
are calculated after a single hydrogen atom is placed between 20 and 50 Å from the
acetone's centre of mass. This is repeated for 30 different random placements of the
hydrogen atom, the mean force difference on the acetone atoms is calculared.

3. Standard deviation in force due to a distance hydrogen

Same as (2), but the standard deviation of the force difference on the acetone atoms is
calculated.


Computational cost
------------------

Low: tests are likely to take less than a minutes to run on CPU.


Data availability
-----------------

None required.


Extensivity
===========

Summary
-------

Performance in respecting extensivity, by measuring differences in energy between
isolated systems, and the same systems combined, but significantly separated.

Metrics
-------

1. Absolute energy difference between isolated and combined slabs

The energy of two isolated slabs is calculated, and the energy of the combined system,
with the two slabs separated by 100 Å is calculated. The absolute energy difference
between the sum of the isolated slabs and that of the combined system is calculated.


Computational cost
------------------

Low: tests are likely to take less than a minutes to run on CPU.


Data availability
-----------------

None required.


Diatomics
=========

Summary
-------

This benchmark probes the short- to medium-range behaviour of every homonuclear and
heteronuclear diatomic pair in the periodic table. Each MLIP is evaluated on a 100-point
linear distance grid spanning 0.18-6.0 Å and the resulting energies and projected forces
are analysed for unphysical oscillations.

Metrics
-------

1. Force flips

   Average number of times the projected bond force changes sign. Forces are projected
   onto the bond axis and values below :math:`10^{-2}` eV/Å are rounded to zero to avoid
   counting noise-induced flips. A smooth curve should switch from attraction to repulsion
   only once at the minimum.


2. Energy minima

   Mean count of distinct minima in the energy-distance profile. Local minima are
   found from the second derivative, where a physical diatomic should show a single
   minimum.


3. Energy inflections

   Mean number of inflection points obtained from the second derivative of the energy
   curve. Inflections are flagged when the second derivative changes sign with a
   tolerance of 0.5 eV/Å² to avoid counting noise-induced inflections. A physical diatomic
   curve should show one inflection point.

4. :math:`\rho(E, \text{repulsion})`

   Spearman correlation between atomic separation and energy on the repulsive side of the well
   (bond lengths ≥ the equilibrium spacing). A perfect diatomic curve should show a strong
   negative correlation, so a value of -1, indicating that as atoms get further apart, the energy
   decreases.

5. :math:`\rho(E, \text{attraction})`

   Spearman correlation between distance and energy on the attractive side (bond lengths
   shorter than the equilibrium spacing). A perfect diatomic curve should show a strong
   positive correlation, so a value of +1, indicating that as atoms get closer together, the
   energy increases.

Computational cost
------------------

High: Expected to take hours to run on GPU, or around one day for slower MLIPs.

Data availability
-----------------

None required; diatomics are generated in ASE.

Iron Properties
===============

Summary
-------

This benchmark evaluates MLIP performance on a comprehensive set of BCC iron
properties relevant to plasticity and fracture. The benchmark is based on
`Zhang et al. (2023) <https://arxiv.org/abs/2307.10072>`_, which assessed the
efficiency, accuracy, and transferability of machine learning potentials for
dislocations and cracks in iron.

Seven groups of properties are computed and compared against DFT (PBE) reference
values: equation of state, elastic constants, the Bain path, vacancy formation
energy, surface energies, generalised stacking fault energies, and
traction-separation curves.

Metrics
-------

EOS properties
^^^^^^^^^^^^^^

1. Lattice parameter error (%)

The equilibrium BCC lattice parameter :math:`a_0` is obtained by fitting a
third-order Birch-Murnaghan equation of state to 30 energy-volume points
sampled around 2.834 Å. The percentage error relative to the DFT reference
value (2.831 Å) is reported.

2. Bulk modulus error (%)

The bulk modulus :math:`B_0` is extracted from the same EOS fit. The
percentage error relative to the DFT reference value (178.0 GPa) is reported.


Elastic constants
^^^^^^^^^^^^^^^^^

3. :math:`C_{11}` error (%)

The elastic constant :math:`C_{11}` is computed using a stress-strain approach
on a 4x4x4 BCC supercell. Small positive and negative strains
(:math:`\pm 10^{-5}`) are applied along each Voigt direction, and the elastic
constants are extracted from the resulting stress differences. The percentage
error relative to the DFT reference value (296.7 GPa) is reported.

4. :math:`C_{12}` error (%)

Same as (3), for the elastic constant :math:`C_{12}`. Reference: 151.4 GPa.

5. :math:`C_{44}` error (%)

Same as (3), for the elastic constant :math:`C_{44}`. Reference: 104.7 GPa.


Vacancy formation energy
^^^^^^^^^^^^^^^^^^^^^^^^^

6. :math:`E_{\mathrm{vac}}` error (%)

A single vacancy is created in a 4x4x4 BCC supercell by removing one atom.
Atomic positions are relaxed at fixed cell volume. The vacancy formation energy
is calculated as
:math:`E_{\mathrm{vac}} = E_{\mathrm{defect}} - E_{\mathrm{perfect}} + E_{\mathrm{coh}}`,
where :math:`E_{\mathrm{coh}}` is the cohesive energy per atom. The percentage
error relative to the DFT reference value (2.02 eV) is reported.


Bain path
^^^^^^^^^

7. BCC-FCC energy difference error (meV)

The Bain path maps the continuous tetragonal distortion from BCC
(:math:`c/a = 1`) to FCC (:math:`c/a = \sqrt{2}`). For each of 65 target
:math:`c/a` ratios between 0.72 and 2.0, a tetragonally distorted cell is
created and its volume is relaxed isotropically (uniform scaling preserving the
:math:`c/a` ratio). The absolute error in the BCC-FCC energy difference
relative to the DFT reference (83.5 meV/atom) is reported.


Surface energies
^^^^^^^^^^^^^^^^

8. Surface energy MAE (J/m²)

Surface energies are computed for the (100), (110), (111), and (112) cleavage
planes. For each surface, a slab is created with vacuum and the surface energy
is calculated as
:math:`\gamma = (E_{\mathrm{slab}} - E_{\mathrm{bulk}}) / 2A`.
Atomic positions are relaxed at fixed cell shape. The mean absolute error
across all four surfaces, relative to DFT reference values
(:math:`\gamma_{100}` = 2.41, :math:`\gamma_{110}` = 2.37,
:math:`\gamma_{111}` = 2.58, :math:`\gamma_{112}` = 2.48 J/m²), is reported.


Stacking fault energies
^^^^^^^^^^^^^^^^^^^^^^^

9. Max SFE :math:`\{110\}\langle111\rangle` error (%)

The generalised stacking fault energy (GSFE) curve for the
:math:`\{110\}\langle111\rangle` slip system is computed by incrementally
displacing the upper half of a crystallographically oriented supercell along
the slip direction. The displacement covers one full Burgers vector
(:math:`b = a\sqrt{3}/2`) in 16 steps. Atoms are constrained to relax only
perpendicular to the fault plane. The percentage error in the maximum
(unstable) SFE relative to the DFT reference (0.75 J/m²) is reported.

10. Max SFE :math:`\{112\}\langle111\rangle` error (%)

Same as (9), for the :math:`\{112\}\langle111\rangle` slip system.
Reference: 1.12 J/m².


Traction-separation curves
^^^^^^^^^^^^^^^^^^^^^^^^^^

11. Max traction (100) error (%)

A traction-separation curve is computed for the (100) cleavage plane by
incrementally separating crystal halves in 0.05 Å steps up to 5.0 Å, without
atomic relaxation, and measuring forces at each step. The traction (tensile
stress) is obtained from the sum of z-forces on the upper region divided by
the cross-sectional area. The percentage error in the maximum traction relative
to the DFT reference (35.0 GPa) is reported.

12. Max traction (110) error (%)

Same as (11), for the (110) cleavage plane. Reference: 30.0 GPa.


Computational cost
------------------

Medium: tests are likely to take minutes to run on GPU, or hours on CPU for each model.
The benchmark is marked as slow and excluded from default test runs.


Data availability
-----------------

Input structures:

* All structures are generated programmatically using ASE. BCC iron unit cells and
  supercells are constructed from the equilibrium lattice parameter obtained via EOS
  fitting.

Reference data:

* DFT (PBE) reference values from:

  * Zhang, L., Csányi, G., van der Giessen, E., & Maresca, F. (2023).
    "Efficiency, Accuracy, and Transferability of Machine Learning Potentials:
    Application to Dislocations and Cracks in Iron."
    `arXiv:2307.10072 <https://arxiv.org/abs/2307.10072>`_
