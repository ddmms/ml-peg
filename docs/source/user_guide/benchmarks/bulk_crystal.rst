=============
Bulk Crystals
=============

Lattice constants
=================

Summary
-------

Performance in evaluating lattice constants for 23 solids, including pure elements,
binary compounds, and semiconductors.


Metrics
-------

1. MAE (Experimental)

Mean lattice constant error compared to experimental data

For each formula, a bulk crystal is built using the experimental lattice constants and
lattice type for the initial structure. This structure is optimised for each model
using the LBFGS optimiser, with the FrechetCellFilter applied to allow optimisation of
the cell, until the largest absolute Cartesian component of any interatomic force is
less than 0.03 eV/Å. The lattice constants of this optimised structure are then
compared to experimental values.


2. MAE (PBE)

Mean lattice constant error compared to PBE data

Same as (1), but optimised lattice constants are compared to reference PBE data.


Computational cost
------------------

Low: tests are likely to less than a minute to run on CPU.


Data availability
-----------------

Input structures:

* Built from experimental lattice constants from various sources

Reference data:

* Experimental data same as input data

* DFT data

  * Batatia, I., Benner, P., Chiang, Y., Elena, A.M., Kovács, D.P., Riebesell, J.,
    Advincula, X.R., Asta, M., Avaylon, M., Baldwin, W.J. and Berger, F., 2025. A
    foundation model for atomistic materials chemistry. The Journal of Chemical
    Physics, 163(18).
  * PBE-D3(BJ)


Elasticity
==========

Summary
-------

Bulk and shear moduli calculated for 12122 bulk crystals from the materials project.


Metrics
-------

(1) Bulk modulus MAE

Mean absolute error (MAE) between predicted and reference bulk modulus (B) values.

MatCalc's ElasticityCalc is used to deform the structures with normal (diagonal) strain
magnitudes of ±0.01 and ±0.005 for ϵ11, ϵ22, ϵ33, and off-diagonal strain magnitudes of
±0.06 and ±0.03 for ϵ23, ϵ13, ϵ12. The Voigt-Reuss-Hill (VRH) average is used to obtain
the bulk and shear moduli from the stress tensor. Both the initial and deformed
structures are relaxed with MatCalc's default ElasticityCalc settings. For more information, see
`MatCalc's ElasticityCalc documentation
<https://github.com/materialsvirtuallab/matcalc/blob/main/src/matcalc/_elasticity.py>`_.

Analysis excludes materials with:
    * B ≤ 0, B > 500 and G ≥ 0, G > 500 structures.
    * H2, N2, O2, F2, Cl2, He, Xe, Ne, Kr, Ar
    * Materials with density < 0.5 (less dense than Li, the lowest density solid element)

(2) Shear modulus MAE

Mean absolute error (MAE) between predicted and reference shear modulus (G) values

Calculated alongside (1), with the same exclusion criteria used in analysis.


Computational cost
------------------

High: tests are likely to take hours-days to run on GPU.


Data availability
-----------------

Input structures:

* 1. De Jong, M. et al. Charting the complete elastic properties of
  inorganic crystalline compounds. Sci Data 2, 150009 (2015).
* Dataset release: mp-pbe-elasticity-2025.3.json.gz from the Materials Project database.

Reference data:

* Same as input data
* PBE



Low-Dimensional Relaxation
==========================

Summary
-------

Performance in relaxing low-dimensional (2D and 1D) crystal structures.
Structures from the Alexandria database are relaxed with cell masks to constrain
relaxation to the appropriate dimensions and compared to PBE reference calculations.


Metrics
-------

**2D Structures:**

(1) Area MAE (2D)

Mean absolute error of area per atom compared to PBE reference.

(2) Energy MAE (2D)

Mean absolute error of energy per atom compared to PBE reference.

(3) Convergence (2D)

Percentage of 2D structures that successfully converged during relaxation.

**1D Structures:**

(4) Length MAE (1D)

Mean absolute error of chain length per atom compared to PBE reference.

(5) Energy MAE (1D)

Mean absolute error of energy per atom compared to PBE reference.

(6) Convergence (1D)

Percentage of 1D structures that successfully converged during relaxation.

Structures are relaxed using janus-core's GeomOpt after calling from `ase.spacegroup.symmetrize.refine_symmetry`. Cell relaxation is constrained using
cell masks:

* 2D: Only in-plane cell components (a, b, and γ) are allowed to relax
* 1D: Only the chain direction (a) is allowed to relax

Relaxation continues until the maximum force component is below 0.0002 eV/Å or until 500 steps are reached. If not converged, relaxation is repeated up to 3 times.


Computational cost
------------------

High: tests are likely to take hours-days to run on GPU, depending on the number of structures tested.


Data availability
-----------------

Input structures:

* Alexandria database 2D structures: https://alexandria.icams.rub.de/data/pbe_2d
* Alexandria database 1D structures: https://alexandria.icams.rub.de/data/pbe_1d
* 3000 structures randomly sampled from each dataset

Reference data:

* Hai-Chen Wang et al 2023 2D Mater. 10 035007
* Jonathan Schmidt et al 2024, Mater. Tod. Phys, 48, 101560
