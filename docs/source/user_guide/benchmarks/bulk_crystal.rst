==================
Bulk Crystals
==================

Elastic Moduli
===

Summary
-------

Bulk and shear moduli calculated for 12122 bulk crystals from the materials project.
MatCalc's ElasticityCalc is used to deform the structures with normal (diagonal) strain
magnitudes of ±0.01 and ±0.005 for ϵ11, ϵ22, ϵ33, and off-diagonal strain magnitudes of
±0.06 and ±0.03 for ϵ23, ϵ13, ϵ12. The Voigt-Reuss-Hill (VRH) average is used to obtain
the bulk and shear moduli from the stress tensor. Both the initial and deformed structures
are relaxed.

Dataset excludes:
    * K <= 0, K > 500 and G <= 0, G > 500 structures.
    * H2, N2, O2, F2, Cl2, He, Xe, Ne, Kr, Ar
    * materials with density < 0.5 (less dense than Li, the least density solid element)

Metrics
-------

Bulk modulus MAE (B)

Mean absolute error (MAE) between predicted and reference bulk modulus values, excluding
values B < -50 GPa and B > 600 GPa.

Shear modulus MAE (G)

Mean absolute error (MAE) between predicted and reference shear modulus values, excluding
values G < -50 GPa and G > 600 GPa.

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
