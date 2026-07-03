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


Bond length distribution
========================

Summary
-------

Performance in maintaining physically reasonable covalent bond lengths during molecular
dynamics of small organic molecules. For each of a set of molecules covering the C-C, C=C,
C#C, C-N, C-O, C=O and C-F bond types, an NVT molecular dynamics simulation is run at 300 K
starting from a QM-optimised reference geometry, and the deviation of a tracked bond from
its reference length is measured along the trajectory.

Metrics
-------

1. Bond length deviation

The length of the tracked bond is measured at each frame of the trajectory, and its absolute
deviation from the reference bond length is averaged over the trajectory and across all
molecules. A well behaved potential keeps bonds close to their reference length, so a lower
deviation is better.

A histogram shows the distribution of the sampled bond length deviations for each model.

Computational cost
------------------

High: one MD simulation per molecule, each 1,000,000 steps. Faster inference can be achieved
using the jax-accelerated simulations in MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. Reference geometries selected from the QM9 dataset
  (Ramakrishnan et al., Scientific Data 1, 140022, 2014).

Reference data:

* QM-optimised equilibrium bond lengths of the reference geometries.
