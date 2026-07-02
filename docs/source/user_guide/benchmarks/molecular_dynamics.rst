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


Water radial distribution
=========================

Summary
-------

Performance in reproducing the oxygen-oxygen radial distribution function of
liquid water. A short NVT molecular dynamics simulation of a box of 500 water
molecules is run from an equilibrated structure, and the resulting O-O RDF is
compared to the experimental reference.

Metrics
-------

1. Peak deviation

The position of the first solvent peak (the radius at which the RDF is maximal)
is compared to the experimental peak of 2.8.

2. RDF RMSE

The root mean square error of the radial distribution function against the
experimental reference, evaluated over the range 2.5-10.0 Å.

A plot shows the predicted RDF profile of each model against the experimental
reference profile.

Computational cost
------------------

High: tests are likely to take several hours on GPU. Faster simulation times can be
achieved using the jax accelerated simulations in MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. Equilibrated box of 500 water
  molecules.

Reference data:

* Experimental oxygen-oxygen radial distribution function.
