====
NEBs
====

Li diffusion
============

Summary
-------

Performance in predicting activation energies of Li diffusion along the [010] and [001]
directions of LiFePO_4.

Metrics
-------

1. [010] (path B) energy barrier error

The initial and final structures for the diffusion of lithium along [010] are created
through deletion an atom from the initial structure. These structures are relaxed,
and the Nudged Elastic Band method is used to calculate the energy barrier. This is
compared to the reference activation energy for this path.


2. [001] (path C) energy barrier error

The initial and final structures for the diffusion of lithium along [001] are created
through deletion an atom from the initial structure. These structures are relaxed,
and the Nudged Elastic Band method is used to calculate the energy barrier. This is
compared to the reference activation energy for this path.

Computational cost
------------------

Medium: tests are likely to take several minutes to run on CPU.


Data availability
-----------------

Input structure:

* Downloaded from Materials Project (mp-19017): https://doi.org/10.17188/1193803

Reference data:

* Manually taken from https://doi.org/10.1149/1.1633511.
* Meta-GGA (Perdew-Wang) exchange correlation functional


Surface reaction
============

Summary
-------

Performance in running NEB for surface reactions on multiple catalytic surfaces from OC20NEB dataset. 

Metrics
-------

1. Convergence 

2. Reaction energy and activation barrier difference compared to DFT reference

3. Sanity of NEB results (in terms of connectivity etc...)

Computational cost
------------------

Medium: tests are likely to take several minutes to run on CPU.
Or Long: 

Data availability
-----------------

Input structure:

* OC20NEB dataset : https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20neb/oc20neb_dft_trajectories_04_23_24.tar.gz

Reference data:

* Manually taken from https://doi.org/10.1021/acscatal.4c04272
* GGA RPBE exchange correlation functional
