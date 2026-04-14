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


Equation of state (metals)
==========================

Summary
-------

Equation of state (energy-volume) curves and phase stability for metals (W, Nb, Mo, Ta, Ti, Zr, Cr, Fe), benchmarked against PBE reference data.

Metrics
-------

1. Δ metric is adapted from `Lejaeghere, K., et. al. Reproducibility in density functional theory
   calculations of solids. (2016). Science, 351(6280), aad3000.
   <https://doi.org/10.1126/science.aad3000>`_. It measures difference between predicted and reference energy-volume curves and is calculated as the square root of the integrated squared energy difference between the predicted and reference curves. It is normalised by the volume range and provides a single number in meV/atom. Here we use it to estimate how well the model reproduces the reference PBE energy-volume curve for the ground state structure of each metal. Note that the volume range is larger than in the original paper, so the Δ metric values are not directly comparable to those reported in the original paper.

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

The reference data  is taken from :

* W, Nb, Mo, Ta `Čák, M., Hammerschmidt, T., Rogal, J., Vitek, V., & Drautz, R. (2014). Analytic bond-order potentials for the bcc refractory metals Nb, Ta, Mo and W. Journal of Physics Condensed Matter, 26(19), 195501. <https://doi.org/10.1088/0953-8984/26/19/195501>`_
* Ti and Zr: `Nitol, M. S., Dickel, D. E., & Barrett, C. D. (2022). Machine learning models for predictive materials science from fundamental physics: An application to titanium and zirconium. Acta Materialia, 224, 117347. <https://doi.org/10.1016/j.actamat.2021.117347>`_
* non-magnetic Cr: `Soulairol, R., Fu, C. C., & Barreteau, C. (2010). Structure and magnetism of bulk Fe and Cr: From plane waves to LCAO methods. Journal of Physics Condensed Matter, 22(29), 295502. <https://doi.org/10.1088/0953-8984/22/29/295502>`_
* ferromagnetic Ni: `He, X., Kong, L. T., & Liu, B. X. (2005). Calculation of ferromagnetic states in metastable bcc and hcp Ni by projector-augmented wave method. Journal of Applied Physics, 97(10). <https://doi.org/10.1063/1.1903104>`_
* ferromegnetic Fe: `Dézerald, L., Marinica, M. C., Ventelon, L., Rodney, D., & Willaime, F. (2014). Stability of self-interstitial clusters with C15 Laves phase structure in iron. Journal of Nuclear Materials, 449(1–3), 219–224. <https://doi.org/10.1016/j.jnucmat.2014.02.012>`_ and `Wang, K., Shang, S. L., Wang, Y., Liu, Z. K., & Liu, F. (2018). Martensitic transition in Fe via Bain path at finite temperatures: A comprehensive first-principles study. Acta Materialia, 147, 261–276. <https://doi.org/10.1016/j.actamat.2018.01.013>`_
