=============
Bulk Crystals
=============

Materials discovery evaluation
==============================

The materials-discovery evaluator computes Matbench Discovery classification and
regression metrics from local tables. The reference table is indexed by
``material_id`` and contains DFT hull distance, DFT formation energy, and a
unique-prototype flag. The prediction table contains ``e_form_per_atom``.

Results are reported for the full test set, unique prototypes, and the 10,000
unique prototypes with the lowest predicted hull distances. They include F1,
discovery acceleration factor (DAF), precision, recall, accuracy, class rates and
counts, MAE, RMSE, R², and missing-prediction counts. Predictions with
formation-energy errors above 5 eV/atom are masked before rounding to three decimal
places. Leaderboard evaluation uses the fraction of unique prototypes with an
unrounded hull distance at or below 0 eV/atom as prevalence, preventing rounding
from changing DAF. Pass this value with ``canonical=True`` and
``uniq_proto_prevalence=...``. Synthetic mode derives prevalence from the rounded
reference values.

See ``ml_peg.analysis.bulk_crystal.materials_discovery``. WBM reference and
prediction artifacts are not included. JSON results include schema and source
framework versions.


Geometry-optimization evaluation
================================

The geo-opt evaluator reads Matbench Discovery-compatible JSONL records with
``material_id``, final ``structure``, ``energy``, ``converged``, and ``n_steps``
fields. It compares relaxed structures with DFT references at symmetry tolerances
of :math:`10^{-2}` and :math:`10^{-5}` by default.

Reported metrics are volume-normalized structure RMSD, symmetry-operation-count
MAE, fractions of symmetry decrease, match, and increase, and the number of valid
symmetry analyses. Structures that cannot be matched receive an RMSD penalty of
1.0. Readers accept plain or gzip-compressed JSONL and CSV files. Per-structure
records are omitted by default to limit memory use on WBM; set
``include_analysis=True`` to return them.

Scalar aggregation is available in the core installation. Structure analysis
requires ``ml-peg[geo-opt]``. No WBM structures, reference analyses, or model
predictions are bundled.


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

(3) Elastic tensor MAE

Element-wise mean absolute error of the 6x6 Voigt-form elasticity tensor.

Symmetry-independent elastic constants are extracted based on crystal system:
triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, or cubic.
Symmetry checks are applied to components on the diagonal with a relative tolerance of 10%. If checks fail, a triclinic symmetry is assumed. Tensor components which are zero in both the reference and comparison tensors are excluded.

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


Energy-volume curves for metals
===============================

Summary
-------

Energy-volume (equation of state) curves and phase stability for metals (W, Nb, Mo, Ta, Ti, Zr, Cr, Fe), benchmarked against PBE reference data.

Metrics
-------

1. Δ metric is adapted from `Lejaeghere, K., et. al. Reproducibility in density functional theory
   calculations of solids. (2016). Science, 351(6280), aad3000.
   <https://doi.org/10.1126/science.aad3000>`_. It measures difference between predicted and reference energy-volume curves and is calculated as the square root of the integrated squared energy difference between the predicted and reference curves. It is normalized by the volume range and provides a single number in meV/atom. Here we use it to extimate how good the model reproduce the reference PBE energy-volume curve for the ground state structure of each metal. Note that the volume range is larger than in the original paper, so the Δ metric values are not directly comparable to those reported in the original paper.

   .. math::

      \Delta = \sqrt{\frac{\int_{V_i}^{V_f} \left[ E_\text{model}(V) - E_\text{ref}(V) \right]^2 \mathrm{d}V}{V_f - V_i}}

   where :math:`E(V)` is the Birch-Murnaghan energy-volume curve and the integral runs over
   a fixed volume range :math:`[V_i, V_f]` around the reference equilibrium volume.
   The result is in meV/atom.

2. Phase energy MAE

   Mean absolute error of the phase energy differences (relative to the ground-state
   phase) evaluated on a uniform volume grid from BM-fitted curves:

   .. math::

      \text{Phase energy MAE} = \frac{1}{N_\phi N_V}
      \sum_{\phi,\, j}
      \left| \Delta E_\text{model}^{\phi}(V_j) - \Delta E_\text{ref}^{\phi}(V_j) \right|

   where :math:`\Delta E^\phi(V) = E^\phi(V) - E^{\phi_0}(V)` is the energy of phase
   :math:`\phi` relative to the ground-state phase :math:`\phi_0`.

   Result in meV/atom. It measures how well the model reproduces the relative energies of different phases across the volume range.

3. Phase stability

   Percentage of volume grid points at which the model correctly identifies all non-ground-state phases as higher in energy than the ground-state phase. A value of 100 % means the model preserves the correct phase ordering at every
   volume point. This metric is practically important: it indicates whether a model predicts spurious phase transitions under extreme conditions, such as the high tensile stresses at a crack tip.

Computational cost
------------------

Low. Every eos curve requires 50 calls on unit cells.

Data availability
-----------------

Input structures:

The perfect bulk crystal cells are created automatically from parsed names from the reference data.

Reference data (PBE):

* W, Nb, Mo, Ta `Čák, M., Hammerschmidt, T., Rogal, J., Vitek, V., & Drautz, R. (2014). Analytic bond-order potentials for the bcc refractory metals Nb, Ta, Mo and W. Journal of Physics Condensed Matter, 26(19), 195501. <https://doi.org/10.1088/0953-8984/26/19/195501>`_
* Ti and Zr: `Nitol, M. S., Dickel, D. E., & Barrett, C. D. (2022). Machine learning models for predictive materials science from fundamental physics: An application to titanium and zirconium. Acta Materialia, 224, 117347. <https://doi.org/10.1016/j.actamat.2021.117347>`_
* non-magnetic Cr: `Soulairol, R., Fu, C. C., & Barreteau, C. (2010). Structure and magnetism of bulk Fe and Cr: From plane waves to LCAO methods. Journal of Physics Condensed Matter, 22(29), 295502. <https://doi.org/10.1088/0953-8984/22/29/295502>`_
* ferromagnetic Ni: `He, X., Kong, L. T., & Liu, B. X. (2005). Calculation of ferromagnetic states in metastable bcc and hcp Ni by projector-augmented wave method. Journal of Applied Physics, 97(10). <https://doi.org/10.1063/1.1903104>`_
* ferromagnetic Fe: `Dézerald, L., Marinica, M. C., Ventelon, L., Rodney, D., & Willaime, F. (2014). Stability of self-interstitial clusters with C15 Laves phase structure in iron. Journal of Nuclear Materials, 449(1–3), 219–224. <https://doi.org/10.1016/j.jnucmat.2014.02.012>`_ and `Wang, K., Shang, S. L., Wang, Y., Liu, Z. K., & Liu, F. (2018). Martensitic transition in Fe via Bain path at finite temperatures: A comprehensive first-principles study. Acta Materialia, 147, 261–276. <https://doi.org/10.1016/j.actamat.2018.01.013>`_


High-Pressure Relaxation
========================

Summary
-------

Performance in relaxing bulk crystal structures under high-pressure conditions.
3000 structures from the Alexandria database are relaxed at 7 pressure conditions
(0, 25, 50, 75, 100, 125, 150 GPa) and compared to PBE reference calculations.


Metrics
-------

For each pressure condition (0, 25, 50, 75, 100, 125, 150 GPa):

(1) Volume MAE

Mean absolute error of volume per atom compared to PBE reference.

(2) Energy MAE

Mean absolute error of enthalpy per atom compared to PBE reference. The enthalpy is
calculated as H = E + PV, where E is the potential energy, P is the applied pressure,
and V is the volume.

(3) Convergence

Percentage of structures that successfully converged during relaxation.

Structures are relaxed using janus-core's GeomOpt with the ase `FixSymmetry` constraint
applied to preserve crystallographic symmetry analogously to DFT. Starting from P000 (0 GPa) structures, each structure is relaxed at the target pressure using the FrechetCellFilter with the
specified scalar pressure. Relaxation continues until the maximum force component is
below 0.0002 eV/Å or until 500 steps are reached. If not converged, relaxation is
repeated up from the last structure of the previous relaxation up to 3 times.


Computational cost
------------------

High: tests are likely to take hours-days to run on GPU, depending on the number of
structures and pressure conditions tested.


Data availability
-----------------

Input structures:

* Alexandria database pressure benchmark dataset
* URL: https://alexandria.icams.rub.de/data/pbe/benchmarks/pressure
* 3000 structures randomly sampled from the full datasets at each pressure

Reference data:

* PBE calculations from the Alexandria database
* Loew et al 2026 J. Phys. Mater. 9 015010 https://iopscience.iop.org/article/10.1088/2515-7639/ae2ba8


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

Mean absolute error of area per atom compared to PBE reference. The area is
calculated as the magnitude of the cross product of the two in-plane lattice vectors.

(2) Energy MAE (2D)

Mean absolute error of energy per atom compared to PBE reference.

(3) Convergence (2D)

Percentage of 2D structures that successfully converged during relaxation.

**1D Structures:**

(4) Length MAE (1D)

Mean absolute error of chain length per atom compared to PBE reference. The length
is the magnitude of the first lattice vector (the chain direction).

(5) Energy MAE (1D)

Mean absolute error of energy per atom compared to PBE reference.

(6) Convergence (1D)

Percentage of 1D structures that successfully converged during relaxation.

Structures are relaxed using janus-core's GeomOpt with the ase `FixSymmetry` constraint
applied to preserve crystallographic symmetry. Cell relaxation is constrained using
cell masks:

* 2D: Only in-plane cell components (a, b, and γ) are allowed to relax
* 1D: Only the chain direction (a) is allowed to relax

Relaxation continues until the maximum force component is below 0.0002 eV/Å or until
500 steps are reached. If not converged, relaxation is repeated up to 3 times.


Computational cost
------------------

High: tests are likely to take hours-days to run on GPU, depending on the number of
structures tested.


Data availability
-----------------

Input structures:

* Alexandria database 2D structures: https://alexandria.icams.rub.de/data/pbe_2d
* Alexandria database 1D structures: https://alexandria.icams.rub.de/data/pbe_1d
* 3000 structures randomly sampled from each dataset

Reference data:

* PBE calculations from the Alexandria database


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


Phonons
=======

Summary
-------

Phonon dispersion and vibrational thermodynamics for 9958 bulk crystals from
the Alexandria phonon benchmark database (a PBE recomputation of the PBEsol
PhononDB/MDR phonon benchmark). The database provides PBE reference
structures, displacements, and force constants. For each MLIP, the structure
is relaxed using the FIRE optimiser with the ``FixSymmetry`` constraint
(fmax = 0.005 eV/Å, up to 1000 steps).
Forces for all displaced supercells are evaluated with the MLIP, and second-
order force constants are calculated and symmetrised on a 6x6x6 q-mesh. Band
structures are computed along the same high-symmetry path as the DFT reference
(101 q-points per segment). Thermal properties are computed on a 20x20x20
q-mesh.

Here, fixing symmetry is important: without it, a structure that
relaxes to a different crystal symmetry would have an incompatible Brillouin
zone path, making metrics such as Avg BZ MAE ill-defined. It also provides a
more stringent test of each MLIP's ability to match the DFT reference.


Metrics
-------

Frequencies are reported in Kelvin-equivalent units (THz x h/k\ :sub:`B`; 1 THz ≈ 48 K).
Thermal properties are evaluated at 300 K.

1. ω\ :sub:`max` MAE (K)

Mean absolute error of the MLIP and DFT maximum phonon frequency across all materials.

2. ω\ :sub:`avg` MAE (K)

Mean absolute error of the MLIP and DFT mean phonon frequency across all materials.

3. ω\ :sub:`min` MAE (K)

Mean absolute error of the MLIP and DFT minimum phonon frequency across all materials.
Because imaginary modes appear as negative values, this metric is sensitive to
whether a model incorrectly predicts dynamical instabilities.

4. S MAE (J/mol·K)

Mean absolute error of MLIP and DFT vibrational entropy at 300 K.

5. F MAE (kJ/mol)

Mean absolute error of the MLIP and DFT vibrational Helmholtz free energy at 300 K.

6. C\ :sub:`V` MAE (J/mol·K)

Mean absolute error of the MLIP and DFT constant-volume heat capacity at 300 K.

7. Avg BZ MAE (K)

Mean MAE of the phonon dispersion across the full Brillouin zone.
The MAE of each MLIP and DFT phonon dispersion branch is computed for a single
material and then the average error over all materials is computed.

8. Stability F1

F1 score for classifying dynamical stability. A material is classified as
stable when ω\ :sub:`min` > -2.4 K (-0.05 THz). Agreement with DFT is
classified as true positive, false positive, true negative, or false negative.


Computational cost
------------------

The DFT reference preprocessing (``test_phonons_ref``) runs once and takes
30 minutes to 1 hour on CPU or GPU, as it only processes pre-computed force
constants from the Alexandria database to generate band structures for later
comparison. This part of the test is marked ``slow``.

The MLIP evaluation (``test_phonons``) requires computing forces for all
displaced supercells of 9958 structures, taking 6-10 hours on GPU. Larger
models will be slower. This part of the test is marked ``very_slow``.


Data availability
-----------------

Input structures and reference force constants:

* Alexandria phonon benchmark database (PBE)
* URL: https://alexandria.icams.rub.de/data/phonon_benchmark/pbe/
* Pre-computed phonopy YAML files containing PBE phonon reference data,
  displacement datasets, force constants, and thermal-property reference values
  for Materials Project structures

Reference data:

* DFT (PBE) force constants, thermal properties and relaxed structures from the Alexandria
  phonon benchmark database.


Elemental TM Vacancy Formation Energies
=======================================

Summary
-------

Performance in predicting vacancy formation energies for 42 fcc or hcp elemental transition metal structures.

Metrics
-------

1. MAE

Mean absolute error (MAE) between predicted and reference (PBE) vacancy formation energies values.

For each elemental transition metal structure, the vacancy formation enthalpy is determined by evaluating the total energy of the cell containing the monovacancy and evaluating the total energy of the undefected cell and comparing these. Note: the undefected cell energy is scaled by a fraction in order to conserve the appropriate number of atoms.

From the reference paper providing the structures, the elemental transition metals are a subset. Note:
    * For magnetic structures, the reference spin-polarized calculations only is used.
    * Unstable element-structure combinations are not included.

Computational cost
------------------

Low: tests are likely to take a couple of minutes to run on CPU.


Data availability
-----------------

Input structures:

* T. Angsten et al. Elemental vacancy diffusion database from high-throughput first-principles calculations for fcc and hcp structures. New J. Phys. 2024 16 015018

Reference data:

* Same as input data
* PBE
