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

4. :math:`\\rho(E, \\text{repulsion})`

   Spearman correlation between atomic separation and energy on the repulsive side of the well
   (bond lengths ≥ the equilibrium spacing). A perfect diatomic curve should show a strong
   negative correlation, so a value of -1, indicating that as atoms get further apart, the energy
   decreases.

5. :math:`\\rho(E, \\text{attraction})`

   Spearman correlation between distance and energy on the attractive side (bond lengths
   shorter than the equilibrium spacing). A perfect diatomic curve should show a strong
   positive correlation, so a value of +1, indicating that as atoms get closer together, the
   energy increases.

Computational cost
------------------

Medium: Expected to take hours to run on GPU, or around one day for slower MLIPs.

Data availability
-----------------

None required; diatomics are generated in ASE.
