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


Si interstitial NEB
===================

Summary
-------

Performance in predicting DFT singlepoint energies and forces along fixed nudged-elastic-band
(NEB) images for a silicon interstitial migration pathway.

Metrics
-------

For each of the three NEB datasets (64 atoms, 216 atoms, and 216 atoms di-to-single), MLIPs are
evaluated on the same ordered NEB images as the reference.

1. Energy MAE

Mean absolute error (MAE) of *relative* energies along the NEB, shifting image 0 to 0 eV
for both the DFT reference and the MLIP predictions.

2. Force MAE

Mean absolute error (MAE) of forces across all atoms and images along the NEB.

Computational cost
------------------

Medium: tests are likely to take several minutes to run on CPU.

Data availability
-----------------

Input/reference data:

* Reference extxyz trajectories (including per-image DFT energies and forces) are distributed as a
  separate zip archive and downloaded on-demand from the ML-PEG data store.
  The calculation script uses the public ML-PEG S3 bucket to retrieve these inputs.
* The reference DFT energies/forces come from Quantum ESPRESSO (PWscf) single-point calculations
  with:

  - Code/version: Quantum ESPRESSO PWSCF v.7.0
  - XC functional: ``input_dft='PBE'``
  - Cutoffs: ``ecutwfc=30.0`` Ry, ``ecutrho=240.0`` Ry
  - Smearing: ``occupations='smearing'``, ``smearing='mv'``, ``degauss=0.01`` Ry
  - SCF convergence/mixing: ``conv_thr=1.0d-6``, ``electron_maxstep=250``, ``mixing_beta=0.2``,
    ``mixing_mode='local-TF'``
  - Diagonalization: ``diagonalization='david'``
  - Symmetry: ``nosym=.false.``, ``noinv=.false.`` (symmetry enabled)
  - Pseudopotential: ``Si.pbe-n-kjpaw_psl.1.0.0.UPF`` (PSLibrary)

  K-points by case:

  - 64 atoms: Γ-only (``K_POINTS automatic 1 1 1 0 0 0``)
  - 216 atoms: Γ-only (``K_POINTS gamma``)
  - 216 atoms di-to-single: Γ-only (``K_POINTS gamma``)
