==================
Molecular Crystals
==================

X23
===

Summary
-------

Performance in predicting lattice energies for 23 molecular crystals.


Metrics
-------

1. Lattice energy error

Accuracy of lattice energy predictions.

For each molecular crystal, lattice energy is calculated by taking the difference
between the energy of the solid molecular crystal divided by the number of molecules it
comprises, and the energy of the isolated molecule. This is compared to the reference
lattice energy.


Computational cost
------------------

Low: tests are likely to take less than a minute to run on CPU.


Data availability
-----------------

Input structures:

* A. M. Reilly and A. Tkatchenko, Understanding the role of vibrations, exact exchange,
  and many-body van der waals interactions in the cohesive properties of molecular
  crystals, The Journal of chemical physics 139 (2013).

Reference data:

* Same as input data
* DMC


DMC-ICE13
=========

Summary
-------

Performance in predicting formation energies of common ice phases.


Metrics
-------

1. Ice phase mean absolute error

Accuracy of formation energy predictions.

For each ice polymorph, the formation energy is calculated by taking the difference
between the energy of the polymorph divided by the number of molecules it comprises,
and the energy of the isolated water molecule. This is compared to the reference
formation energy.


Computational cost
------------------

Low: tests are likely to take less than a minute to run on CPU.


Data availability
-----------------

Input structures:

* F. Della Pia, A. Zen, D. Alfè, and A. Michaelides, Dmc-ice13: Ambient and high
  pressure polymorphs of ice from diffusion monte carlo and density functional theory,
  The Journal of Chemical Physics 157 (2022).

Reference data:

* Same as input data
* PBE-D3(BJ)


CPOSS209
=========

Summary
-------

Performance in predicting lattice energies of 209 organic molecular crystals from the CPOSS209 dataset.


Metrics
-------

1. Absolute lattice energy MAE

Accuracy of the absolute lattice energy predictions.

For each molecular crystal, lattice energy is calculated by taking the difference between the energy of the solid molecular crystal divided by the number of molecules it comprises,
and the energy of the isolated molecule. This is compared to the reference lattice energy.

2. Relative lattice energy MAE

Accuracy of the reltive lattice energy predictions.

We compute the lattice energies in the same way as before, but this time we compute the relative error.
The relative error is computed by identifying the most stable polymorphs and, for each crystal, subtracting the lattice energy of the most stable polymorph.
This metric cares more about the ranking of the crystal polymorphs rather than reproducing the reference values of the lattice energies.


Computational cost
------------------

Low: tests should take a few minutes on a CPU


Data availability
-----------------

Input structures:

* Structures are optimized at the PBE+TS level.
* Louise S. Price, Matteo Paloni, Matteo Salvalaglio, and Sarah L. Price
  Crystal Growth & Design 2025 25 (9), 3186-3209
  DOI: 10.1021/acs.cgd.5c00255

Reference data:

* Same as input data
* wb97md3 with 1b CCSD(T) corrections


MC500
=====

Summary
-------

Performance in reproducing 500 r2SCAN+MBD molecular-crystal geometries. Atomic
positions and unit-cell parameters are relaxed while preserving the starting space-group
symmetry.


Metrics
-------

1. Mean RMS Cartesian displacement excluding hydrogen

The root-mean-square Cartesian displacement (RMSCD) compares the fractional-coordinate
displacements using both the reference and relaxed cells. Hydrogen atoms are excluded
from the primary metric because their experimental positions are generally less reliable.

2. Structures within 0.25 Å

The percentage of the complete dataset with a non-hydrogen RMSCD no greater than the
published 0.25 Å threshold. Failed calculations therefore do not count as passing.

3. Convergence

The percentage of structures that reach a maximum force below 0.01 eV/Å within 1000
BFGS optimization steps.

The all-atom RMSCD is also reported for context but is excluded from the overall score.


Benchmark speed
---------------

Slow: the benchmark performs symmetry-constrained variable-cell relaxations of 500
molecular crystals, with up to 1000 optimization steps for each structure, taking
multiple hours on GPU for each model.


Data availability
-----------------

Input and reference structures:

* 500 molecular-crystal structures optimized with r2SCAN+MBD using CASTEP.
* M. Husak, F. Fnukal and J. Cejka, *The use of machine learning interatomic
  potentials for the verification of experimental molecular crystal structures*,
  IUCrJ **13**, 623-634 (2026), DOI: 10.1107/S2052252526006949.

RMSCD definition:

* J. van de Streek and M. A. Neumann, *Validation of experimental molecular crystal
  structures with dispersion-corrected density functional theory calculations*,
  Acta Crystallographica Section B **66**, 544-558 (2010).
