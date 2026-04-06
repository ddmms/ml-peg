======
Carbon
======

Diverse Carbon Structures
=========================

Summary
-------

Performance in reproducing PBE energies and forces across a diverse set of pure carbon
structures covering major bonding environments. The dataset was constructed by
Bochkarev et al. to train and validate an Atomic Cluster Expansion (ACE) potential
for carbon, and spans 17,293 structures with 366,763 atoms in total.

The benchmark reports energy-per-atom MAE separately for each of the five structural
categories in the dataset, enabling targeted assessment of model performance across
sp²-bonded, sp³-bonded, amorphous/liquid, general bulk, and general cluster
environments.

Dataset composition (Table 1 from Bochkarev et al.):

.. list-table::
   :header-rows: 1
   :widths: 20 45 10 10 20 15

   * - Category
     - Description
     - Structures
     - Atoms
     - Energy range (eV/atom)
     - NNB range (Å)
   * - sp² bonded
     - Graphene, graphite, fullerenes, nanotubes, incl. defects
     - 3532
     - 88,358
     - [-9.07, 78.50]
     - [0.7, 4.4]
   * - sp³ bonded
     - Cubic and hexagonal diamond, high-pressure phases (bc8, st12, m32, etc.), incl. defects
     - 3407
     - 84,290
     - [-8.93, 36.99]
     - [0.9, 4.9]
   * - amorphous/liquid
     - Selected from available datasets; amorphous and liquid phases, MD trajectories of multilayered graphene
     - 2642
     - 146,188
     - [-9.06, -3.18]
     - [1.0, 1.7]
   * - general bulk
     - Basic crystals: fcc, hcp, bcc, sc, A15, etc. over a broad range of volumes and random displacements/cell deformations
     - 5342
     - 39,126
     - [-8.06, 82.17]
     - [0.9, 4.4]
   * - general clusters
     - Non-periodic clusters with 2-6 atoms
     - 2370
     - 8,801
     - [-6.19, 83.28]
     - [0.6, 5.0]

Metrics
-------

Energy (eV/atom MAE)
^^^^^^^^^^^^^^^^^^^^

1. sp² bonded MAE

   Mean absolute error of predicted vs. reference total energy per atom for all
   sp²-bonded structures (graphene, graphite, fullerenes, nanotubes, and their
   defect variants).

2. sp³ bonded MAE

   Mean absolute error of predicted vs. reference total energy per atom for all
   sp³-bonded structures (cubic and hexagonal diamond, high-pressure polymorphs
   including bc8, st12, m32, and their defect variants).

3. amorphous/liquid MAE

   Mean absolute error of predicted vs. reference total energy per atom for
   amorphous and liquid carbon configurations selected from MD trajectories,
   including multilayered graphene.

4. general bulk MAE

   Mean absolute error of predicted vs. reference total energy per atom for
   general bulk crystal structures (fcc, hcp, bcc, sc, A15, etc.) sampled
   over a broad range of volumes and cell deformations.

5. general clusters MAE

   Mean absolute error of predicted vs. reference total energy per atom for
   non-periodic carbon clusters containing 2–6 atoms.

Computational cost
------------------

Low: single-point calculations on all 17,293 structures will complete in minutes
on GPU or CPU

Data availability
-----------------

Input structures and reference data:

*
Reference data:

* Reference energies and forces were computed with highly converged PBE-DFT
   using the PAW method for carbon, a 500 eV plane-wave cutoff, 10\ :sup:`−6` eV
   electronic convergence, and 0.1 eV Gaussian smearing. Periodic systems used
   dense Γ-centered k-point meshes with spacing 0.125 Å\ :sup:`−1`, while
   non-periodic clusters were sampled at the Γ point only. An additional support
   grid was used to reduce force noise.
