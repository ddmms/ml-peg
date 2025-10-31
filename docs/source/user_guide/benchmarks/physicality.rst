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
