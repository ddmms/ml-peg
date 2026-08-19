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

Si defects
==========

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

Oxygen adatom diffusion on 2D TMDs
==================================

Summary
-------

Performance in predicting diffusion barriers for oxygen adatom migration on monolayer 2D transition metal dichalcogenide (TMD) surfaces.

Metrics
-------

1. Diffusion barrier error

The diffusion barrier is defined as the maximum energy along the NEB minimum energy path (MEP), relative to the initial state energy. This is compared to the reference DFT-PBE calculated barrier for each material.

The benchmark includes the following 2D TMD materials (2H phase):

* MoS₂
* MoSe₂
* MoTe₂
* WS₂
* WSe₂
* WTe₂

Computational cost
------------------

Medium: tests involving multiple foundation models are likely to take around an one hour to run on a single GPU.

Data availability
-----------------

Input structures:

* Primitive cells can downloaded from the Materials Project (2H bulk structures):

  - MoS₂: mp-2815 https://legacy.materialsproject.org/materials/mp-2815/
  - MoSe₂: mp-1634 https://legacy.materialsproject.org/materials/mp-1634/
  - MoTe₂: mp-602 https://legacy.materialsproject.org/materials/mp-602/
  - WS₂: mp-224 https://legacy.materialsproject.org/materials/mp-224/
  - WSe₂: mp-1821 https://legacy.materialsproject.org/materials/mp-1821/
  - WTe₂: mp-1019322 https://legacy.materialsproject.org/materials/mp-1019322/

* Chalcogen-chalcogen thicknesses extracted from these bulk structures by computing intra-layer vertical separations.
* In-plane lattice constants manually extracted from Liu et al. (see reference data below):
    - MoS₂ 3.183 Å, MoSe₂ 3.318 Å, MoTe₂ 3.547 Å, WS₂ 3.182 Å, WSe₂ 3.317 Å, WTe₂ 3.551 Å.
* Monolayer 6x6x1 supercells built using ASE ``build.mx2`` (https://wiki.fysik.dtu.dk/ase/ase/build/surface.html#ase.build.mx2) using the lattice constants and thicknesses reported here.
* End pairs created by placing an oxygen adatom in adjacent sites with initial heights of around 1.5-1.7 Å depending on the chalcogen atom.
* Structures are then relaxed by the foundation model being tested.

Reference data:

* Manually taken from https://doi.org/10.1039/C4RA17320A
* GGA (PBE) exchange correlation functional
