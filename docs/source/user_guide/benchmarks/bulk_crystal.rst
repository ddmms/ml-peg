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


Diamond phonons
===============

Summary
-------

Performance in predicting the phonon dispersion of bulk diamond (carbon).

The benchmark evaluates the accuracy of phonon frequencies along a fixed
high-symmetry path in the Brillouin zone for a single reference crystal
structure of diamond.


Metrics
-------

1. Band MAE

Mean absolute error (MAE) between predicted and reference phonon frequencies.

For bulk diamond, the phonon band structure is computed for each model along the same
q-point path as the reference calculation. At each q-point, the six phonon frequencies
are compared to the reference frequencies after sorting the modes to avoid branch
labelling ambiguities. The MAE is evaluated over all q-points and all phonon branches.


2. Band RMSE

Root mean squared error (RMSE) between predicted and reference phonon frequencies.

The RMSE is computed using the same sorted, mode-unlabelled comparison procedure as in
(1), over all q-points and all phonon branches.


Computational cost
------------------

Medium: tests typically take a few minutes to run on CPU.


Data availability
-----------------

Input structures:

* A primitive bulk diamond unit cell containing two carbon atoms, used to generate a
  phonopy displacement dataset on a 4×4×4 supercell.

Reference data:

* DFT phonon band structure for bulk diamond along a fixed high-symmetry path, provided
  as ``dft_band.npz``.
* RSCAN.
