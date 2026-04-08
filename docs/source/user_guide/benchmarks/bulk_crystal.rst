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


Split vacancy
=============

Summary
-------

Performance predicting split vacancy formation energies and relaxed structures for
metal oxides (PBEsol, 531 host compounds, 722 material-cation pairs, 2154 structures)
and stable nitrides (PBE, 144 host compounds, 149 material-cation pairs, 285 structures).

A split vacancy is a vacancy defect that does not reside at a normal cation lattice site
but instead at an interstitial position. These are distinct from simple vacancies and
their prediction requires an MLIP to accurately resolve the energy landscape between
competing defect geometries.

Data from Seán Kavanagh, *Identifying split vacancy defects with machine-learned foundation models and electrostatics*,
`https://doi.org/10.1088/2515-7655/ade916 <https://doi.org/10.1088/2515-7655/ade916>`_

Metrics
-------

1. MAE (formation energy)

Mean absolute error of the split vacancy formation energy, defined as the energy
difference between the lowest-energy relaxed split vacancy (SV) and normal vacancy (NV)
structures:

.. math::

   E_\text{form} = \min_i E^\text{SV}_i - \min_j E^\text{NV}_j

where the minima are taken over all initial structures relaxed for a given
material-cation pair.

2. Spearman's rank correlation

For each material-cation pair, all DFT-relaxed NV and SV structures are collected.
The Spearman's rank correlation is computed between the DFT total energies of these
structures and the corresponding MLIP single-point energies (evaluated at the
DFT-relaxed geometries). A high coefficient indicates the MLIP correctly orders the
relative energies of the competing defect configurations. The mean coefficient across
all material-cation pairs is reported.

3. Match Rate

Fraction of MLIP-relaxed structures that converge to the same geometry as the
DFT-relaxed reference, determined using the pymatgen
`StructureMatcher <https://pymatgen.org/pymatgen.analysis.html#module-pymatgen.analysis.structure_matcher>`_.
The match criterion is a normalised maximum atomic displacement below 0.3 (see
metric 4).

4. Max Dist

Maximum atomic displacement between the MLIP-relaxed and DFT-relaxed matched
structures, normalised by :math:`(N/V)^{1/3}` (where :math:`N` is the number of
atoms and :math:`V` the cell volume) to give a unitless quantity comparable across
different supercell sizes. Only computed for structure pairs that pass the
StructureMatcher test. The match criterion itself is a normalised max dist below 0.3.


Computational cost
------------------

Relatively slow: relaxations involve large defect supercells (50–500 atoms) and
multiple initial structures per material-cation pair.


Data availability
-----------------

Input structures:

* Generated using the `doped <https://github.com/SMTG-Bham/doped>`_ supercell
  algorithm. For oxides, supercell parameters are consistent with those of
  Kumagai et al. (using the ``vise`` package). Supercells satisfy a minimum image
  distance of 10 Å and a minimum of 50 atoms.

Reference data:

* See: Seán Kavanagh, *Identifying split vacancy defects with machine-learned foundation models and electrostatics*,
`https://doi.org/10.1088/2515-7655/ade916 <https://doi.org/10.1088/2515-7655/ade916>`_
* All DFT calculations performed with VASP using PAW pseudopotentials. Oxides:
  PBEsol functional, 400 eV plane-wave cutoff, Γ-point sampling, 0.01 eV Å⁻¹
  force convergence. Nitrides: PBE functional with MPRelaxSet parameters,
  520 eV plane-wave cutoff. See paper for full details.
