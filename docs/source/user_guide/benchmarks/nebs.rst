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

Performance in running NEB for three surface reactions (adsorption/desorption, dissociation, transfer) from OC20NEB dataset.

Metrics
-------

1. Activation barrier error

Initial and final geometries are from OC20NEB dataset and relaxed with each model with several layers of slab fixed. And interpolation is generated with 10 images including initial and final images. Following NEB setting from original paper, initially it runs without climbing image until fmax=0.45 eV/A or max steps 200 and converts to climbing image mode with fmax=0.05 eV/A or max steps 300. Barrier is measured between the highest energy point and initial image.

The benchmark includes following reactions:

- desorption_ood_87_9841_0_111-1
- dissociation_ood_268_6292_46_211-5
- transfer_id_601_1482_1_211-5

Computational cost
------------------

Slow: tests are likely to take more than 2 hours to run on single GPU.

Data availability
-----------------

Input structure:

* OC20NEB dataset : https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20neb/oc20neb_dft_trajectories_04_23_24.tar.gz

Reference data:

* Manually taken from https://doi.org/10.1021/acscatal.4c04272
* GGA RPBE exchange correlation functional
