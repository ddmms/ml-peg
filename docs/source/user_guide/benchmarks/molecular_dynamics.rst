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


Stability
=========

Summary
-------

Performance in running stable molecular dynamics across a diverse set of
systems: small molecules (containing H/C/N/O, sulfur, and halogens), peptides
in vacuum (neurotensin, PDB: 2LNF; cyclic oxytocin, PDB: 7OFG), a protein in
vacuum (PDB: 1A7M), and solvated peptides with and without counter-ions.

Metrics
-------

1. Success rate

Each system is run with a short NVT molecular dynamics simulation at 300 K. The
fraction of simulations that complete without error is reported. A simulation
counts as failed if it raises an error during integration (for example, a
numerical instability). Higher is better, with 1.0 meaning every system ran to
completion.

Computational cost
------------------

High: tests are likely to take many hours on GPU. Faster simulation times can be
achieved using the jax accelerated simulations in MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep.
  Structures derived from PDB entries 2LNF, 7OFG, and 1A7M.
