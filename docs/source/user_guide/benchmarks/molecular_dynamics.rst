==================
Molecular dynamics
==================

Liquid densities
================

Summary
-------

Performance in predicting densities for 61 organic liquids, each system consisting of
about 1000 atoms. The dataset covers aliphatic, aromatic molecules, as well as different
functional groups and halogenated molecules.

Metrics
-------

1. Density error

For each system, the density is calculated by taking the average density of an NPT molecular
dynamics run. The initial part of the simulation, here 500 ps, is omitted from the density
calculation. This is compared to the reference density, obtained from experiment.

Computational cost
------------------

Very high: tests are likely to take several days to run on GPU.

Data availability
-----------------

Input structures:

* Weber et al., Efficient Long-Range Machine Learning Force Fields for
    Liquid and Materials Properties.
    arXiv:2505.06462 [physics.chem-ph]

Reference data:

* Same as input data
* Experimental


Polymer densities
=================

Summary
-------

Performance in predicting room-temperature amorphous densities for 130
polymers. Starting structures are prebuilt polymer cells, and the reference
data are experimental densities.

Metrics
-------

1. Density error

For each polymer, the density is calculated by averaging over the final 500 ps
NPT production stage of a 24-stage Polymatic-style equilibration protocol. The
protocol uses a 0.5 fs timestep and runs for about 2.06 ns per model and
polymer at the default time prefactor. The predicted density is compared to the
experimental reference density.

Computational cost
------------------

Very high: tests are likely to take several days to run on GPU.

Data availability
-----------------

Input structures:

* Built from the SimPoly polymer dataset and stored as prebuilt structures for
  ML-PEG calculations.

Reference data:

* Simm et al., SimPoly: Simulation of Polymers with Machine Learning Force
  Fields Derived from First Principles. arXiv:2510.13696 [physics.comp-ph]
* Experimental


Water density
=============

Summary
-------

Performance in predicting the density of water at temperatures of 270, 290, 300, and 330 K.
The water systems consist of 333 molecules.

Metrics
-------

1. Density error

For each system, the density is calculated by taking the average density of an NPT molecular
dynamics run. The initial part of the simulation, here 500 ps, is omitted from the density
calculation. This is compared to the reference density, obtained from experiment.

Computational cost
------------------

Very high: tests are likely to take several days to run on GPU.

Data availability
-----------------

Input structures:

* Weber et al., Efficient Long-Range Machine Learning Force Fields for
  Liquid and Materials Properties. arXiv:2505.06462 [physics.chem-ph]

Reference data:

* Same as input data
* Experimental
