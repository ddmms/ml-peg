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


Iron Properties
===============

Summary
-------

This benchmark evaluates MLIP performance on a comprehensive set of BCC iron
properties relevant to plasticity and fracture. The benchmark is based on
`Zhang et al. (2023) <https://arxiv.org/abs/2307.10072>`_, which assessed the
efficiency, accuracy, and transferability of machine learning potentials for
dislocations and cracks in iron.

Seven groups of properties are computed and compared against DFT (PBE) reference
values: equation of state, elastic constants, the Bain path, vacancy formation
energy, surface energies, generalised stacking fault energies, and
traction-separation curves.

Metrics
-------

EOS properties
^^^^^^^^^^^^^^

1. Lattice parameter error (%)

The equilibrium BCC lattice parameter :math:`a_0` is obtained by fitting a
third-order Birch-Murnaghan equation of state to 30 energy-volume points
sampled around 2.834 Å. The percentage error relative to the DFT reference
value (2.831 Å) is reported.

2. Bulk modulus error (%)

The bulk modulus :math:`B_0` is extracted from the same EOS fit. The
percentage error relative to the DFT reference value (178.0 GPa) is reported.


Elastic constants
^^^^^^^^^^^^^^^^^

3. :math:`C_{11}` error (%)

The elastic constant :math:`C_{11}` is computed using a stress-strain approach
on a 4x4x4 BCC supercell. Small positive and negative strains
(:math:`\pm 10^{-5}`) are applied along each Voigt direction, and the elastic
constants are extracted from the resulting stress differences. The percentage
error relative to the DFT reference value (296.7 GPa) is reported.

4. :math:`C_{12}` error (%)

Same as (3), for the elastic constant :math:`C_{12}`. Reference: 151.4 GPa.

5. :math:`C_{44}` error (%)

Same as (3), for the elastic constant :math:`C_{44}`. Reference: 104.7 GPa.


Vacancy formation energy
^^^^^^^^^^^^^^^^^^^^^^^^^

6. :math:`E_{\mathrm{vac}}` error (%)

A single vacancy is created in a 4x4x4 BCC supercell by removing one atom.
Atomic positions are relaxed at fixed cell volume. The vacancy formation energy
is calculated as
:math:`E_{\mathrm{vac}} = E_{\mathrm{defect}} - E_{\mathrm{perfect}} + E_{\mathrm{coh}}`,
where :math:`E_{\mathrm{coh}}` is the cohesive energy per atom. The percentage
error relative to the DFT reference value (2.02 eV) is reported.


Bain path
^^^^^^^^^

7. BCC-FCC energy difference error (meV)

The Bain path maps the continuous tetragonal distortion from BCC
(:math:`c/a = 1`) to FCC (:math:`c/a = \sqrt{2}`). For each of 65 target
:math:`c/a` ratios between 0.72 and 2.0, a tetragonally distorted cell is
created and its volume is relaxed isotropically (uniform scaling preserving the
:math:`c/a` ratio). The absolute error in the BCC-FCC energy difference
relative to the DFT reference (83.5 meV/atom) is reported.


Surface energies
^^^^^^^^^^^^^^^^

8. Surface energy MAE (J/m²)

Surface energies are computed for the (100), (110), (111), and (112) cleavage
planes. For each surface, a slab is created with vacuum and the surface energy
is calculated as
:math:`\gamma = (E_{\mathrm{slab}} - E_{\mathrm{bulk}}) / 2A`.
Atomic positions are relaxed at fixed cell shape. The mean absolute error
across all four surfaces, relative to DFT reference values
(:math:`\gamma_{100}` = 2.41, :math:`\gamma_{110}` = 2.37,
:math:`\gamma_{111}` = 2.58, :math:`\gamma_{112}` = 2.48 J/m²), is reported.


Stacking fault energies
^^^^^^^^^^^^^^^^^^^^^^^

9. Max SFE :math:`\{110\}\langle111\rangle` error (%)

The generalised stacking fault energy (GSFE) curve for the
:math:`\{110\}\langle111\rangle` slip system is computed by incrementally
displacing the upper half of a crystallographically oriented supercell along
the slip direction. The displacement covers one full Burgers vector
(:math:`b = a\sqrt{3}/2`) in 16 steps. Atoms are constrained to relax only
perpendicular to the fault plane. The percentage error in the maximum
(unstable) SFE relative to the DFT reference (0.75 J/m²) is reported.

10. Max SFE :math:`\{112\}\langle111\rangle` error (%)

Same as (9), for the :math:`\{112\}\langle111\rangle` slip system.
Reference: 1.12 J/m².


Traction-separation curves
^^^^^^^^^^^^^^^^^^^^^^^^^^

11. Max traction (100) error (%)

A traction-separation curve is computed for the (100) cleavage plane by
incrementally separating crystal halves in 0.05 Å steps up to 5.0 Å, without
atomic relaxation, and measuring forces at each step. The traction (tensile
stress) is obtained from the sum of z-forces on the upper region divided by
the cross-sectional area. The percentage error in the maximum traction relative
to the DFT reference (35.0 GPa) is reported.

12. Max traction (110) error (%)

Same as (11), for the (110) cleavage plane. Reference: 30.0 GPa.


Computational cost
------------------

Medium: tests are likely to take minutes to run on GPU, or hours on CPU for each model.
The benchmark is marked as slow and excluded from default test runs.


Data availability
-----------------

Input structures:

* All structures are generated programmatically using ASE. BCC iron unit cells and
  supercells are constructed from the equilibrium lattice parameter obtained via EOS
  fitting.

Reference data:

* DFT (PBE) reference values from:

  * Zhang, L., Csányi, G., van der Giessen, E., & Maresca, F. (2023).
    "Efficiency, Accuracy, and Transferability of Machine Learning Potentials:
    Application to Dislocations and Cracks in Iron."
    `arXiv:2307.10072 <https://arxiv.org/abs/2307.10072>`_
