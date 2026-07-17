=======
Defects
=======
Defectstab
==========

Summary
-------

This benchmark evaluates formation energies across four distinct subsets covering
different material systems: iron self-interstitial atoms (SIA), boron carbide
stoichiometry, boron carbide point defects, and a methylammonium lead iodide
vacancy.
Results can be viewed per subset in the app. The subsets are described in detail below.


fe_sia
^^^^^^

Formation energies of 5 single self-interstitial atom configurations in a
128-atom BCC iron supercell.

The formation energy is calculated as:

.. math::
    E_f = E_{\mathrm{config}} - \frac{N_{\mathrm{config}}}{N_{\mathrm{bulk}}} E_{\mathrm{bulk}}

where :math:`E_{\mathrm{config}}` is the total energy of the interstitial configuration
containing :math:`N_{\mathrm{config}}` atoms, and :math:`E_{\mathrm{bulk}}` is the energy of the
perfect bulk supercell consisting of :math:`N_{\mathrm{bulk}}` atoms.

* DFT reference: PBE exchange-correlation functional.

boroncarbide_stoichiometry
^^^^^^^^^^^^^^^^^^^^^^^^^^

Formation enthalpies of 6 boron carbide phases with varying stoichiometries
(:math:`\mathrm{B_x C}` with :math:`4 \leq x \leq 10.5`).

The formation enthalpy is calculated as:

.. math::
    H_f = E_{\mathrm{phase}} - n_\mathrm{B} \, E_\mathrm{B} - n_\mathrm{C} \, E_\mathrm{C}

where :math:`n_\mathrm{B}` and :math:`n_\mathrm{C}` are the number of boron and carbon atoms in
the structure, :math:`E_\mathrm{B} = E(\alpha \text{-} \mathrm{boron})/12` and
:math:`E_\mathrm{C} = E(\mathrm{graphite})/4` are the per-atom reference energies of the
elements.

* DFT reference: LDA exchange-correlation functional.

boroncarbide_defects
^^^^^^^^^^^^^^^^^^^^

Formation energies of 3 point defects in :math:`\mathrm{B_4C}` boron carbide (boron-rich
conditions): a bipolar defect (B/C exchange on a :math:`\mathrm{B_{11}C}` icosahedron)
and two variants of a chain boron vacancy (VB0 and VB0_CC).

The formation energies are:

* **Bipolar defect:**

  .. math::
      E_f = E_{\mathrm{Defect}} - E_{\mathrm{NoDefects}}

* **Boron vacancy (B-rich):**

  .. math::
      E_f = E_{\mathrm{Defect}} - E_{\mathrm{NoDefects}} + \mu_\mathrm{B}

  where :math:`\mu_\mathrm{B} = E_\mathrm{B}` for boron-rich conditions,
  and :math:`\mu_\mathrm{B} = (E_{\mathrm{NoDefects}}/24 - E_\mathrm{C})/4` for carbon-rich conditions.
  The current benchmark uses **boron-rich** conditions.

* DFT reference: LDA exchange-correlation functional.

mapi_tetragonal
^^^^^^^^^^^^^^^

Formation energy of a methylammonium + iodine divacancy (:math:`\mathrm{VMAI}`) in a
16-formula-unit supercell of tetragonal :math:`\mathrm{MAPbI_3}`.

The formation energy is calculated as:

.. math::
    E_f = E(\mathrm{VMAI}) + \frac{1}{2} E(\mathrm{MAI}) - E_{\mathrm{pristine}}

where :math:`E(\mathrm{VMAI})` is the energy of the supercell with the divacancy,
:math:`E(\mathrm{MAI})` is the energy of the methylammonium iodide molecular crystal,
and :math:`E_{\mathrm{pristine}}` is the energy of the pristine supercell.

* DFT reference: optB86b+vdW exchange-correlation functional.

Metrics
-------

**RMSD**

Root Mean Square Deviation of formation energies compared to DFT data.
The RMSD of formation energies with respect to DFT data is computed independently
for each subset. Per-subset values are reported as separate columns in the app table.

For each subset, the ``bad`` threshold is set as:

.. math::
  \mathrm{bad} = 0.5 \times \mathrm{mean}\left(|E_\mathrm{ref}|\right)

where :math:`\mathrm{mean}\left(|E_\mathrm{ref}|\right)` is the mean of the absolute DFT
reference formation energies in that subset. The ``good`` threshold is 0 eV (perfect agreement).

Each per-subset RMSD is converted to a normalised score in :math:`[0, 1]` by linear
interpolation between ``good`` (score = 1) and ``bad`` (score = 0). If the RMSD
exceeds the ``bad`` threshold, the subset score is clamped to 0. The overall
``Score`` reported in the table is the unweighted average of all four per-subset
scores.

Computational cost
------------------

Low: The geometries are static, requiring only single-point energy calculations
for the configurations and reference structures.

Data availability
-----------------

Input structures:

* Subset ``fe`` of the ``Defectstab`` dataset (:math:`\mathrm{Fe}` SIA configurations, PBE functional).

  * A. Allera, T.D. Swinburne, A.M. Goryaeva, B. Bienvenu, F. Ribeiro, M. Perez, M.-C. Marinica, D. Rodney,
    Activation entropy of dislocation glide in body-centered cubic metals from atomistic simulations,
    Nat Commun 16, 8367 (2025).

  * A.M. Goryaeva, C. Lapointe, C. Dai, J. Dérès, J.-B. Maillet, M.-C. Marinica,
    Reinforcing materials modelling by encoding the structures of defects in crystalline solids into distortion scores,
    Nat Commun 11, 4691 (2020).

* Subset ``boroncarbide_stoichiometry`` and ``boroncarbide_defects`` of the ``Defectstab`` dataset (Boron Carbide structures, LDA functional).

  * G. Roma, K. Gillet, A. Jay, N. Vast, G. Gutierrez,
    Understanding first-order Raman spectra of boron carbides across the homogeneity range,
    Phys. Rev. Materials 5, 063601 (2021).

* Subset ``mapi_tetragonal`` of the ``Defectstab`` dataset (MAPI tetragonal structures, optB86b+vdW functional).

  * K. Madaan, G. Roma, J. Gulomov, P. Pochet, C. Corbel, I. Makkonen,
    Challenges in predicting positron annihilation lifetimes in lead halide perovskites: correlation functionals and polymorphism,
    arXiv:2511.06926 (2025).

  * K. Madaan,
    Phases and vacancy defects in methylammonium lead iodide perovskite: an ab initio study,
    PhD thesis, Université Paris-Saclay (2023).

Reference data:

* Computed from the DFT total energies provided with the input structures.



Relastab
========

Summary
-------

This benchmark evaluates the ability of models to correctly identify the most
stable interstitial configuration and to correctly rank the least stable ones.
The evaluation is performed across multiple subsets representing different host
systems (:math:`\mathrm{Fe}` and :math:`\mathrm{CaWO_4}`), and the final scores are averaged over
these subsets.

* DFT reference: PBE exchange-correlation functional.

Metrics
-------

Two metrics are evaluated per subset; per-subset values are reported as separate
columns in the app table.

**GlobalMin**
  Binary score (1 if the predicted lowest-energy configuration matches the
  reference global minimum, 0 otherwise).

**Top5 Spearman**
  Spearman rank correlation between predicted and reference rankings for the 5
  highest-energy (least stable) configurations in the subset.

For both metrics the ``good`` threshold is 1.0 (perfect) and the ``bad``
threshold is 0.0. Each per-subset metric value is therefore used directly as its
normalised score in :math:`[0, 1]`. The overall ``Score`` reported in the table
is the unweighted average of all per-subset metric scores across both metrics
and both subsets.

Computational cost
------------------

Low: Requires single-point energy calculations for the configurations in each
subset.

Data availability
-----------------

Input structures:

* Subset ``fe`` of the ``Relastab`` dataset (:math:`\mathrm{Fe}` SIA configurations, PBE functional).

  * A. Allera, T.D. Swinburne, A.M. Goryaeva, B. Bienvenu, F. Ribeiro, M. Perez, M.-C. Marinica, D. Rodney,
    Activation entropy of dislocation glide in body-centered cubic metals from atomistic simulations,
    Nat Commun 16, 8367 (2025).

  * A.M. Goryaeva, C. Lapointe, C. Dai, J. Dérès, J.-B. Maillet, M.-C. Marinica,
    Reinforcing materials modelling by encoding the structures of defects in crystalline solids into distortion scores,
    Nat Commun 11, 4691 (2020).

* Subset ``cawo4`` of the ``Relastab`` dataset (:math:`\mathrm{CaWO_4}` interstitial configurations, PBE functional).

  * G. Soum-Sidikov, A. Boisard, D. L010, M. Loidl, O. Stézowski, A. Music, G. Music, Y. Zeng, R. Cong, T. Yang, A. Echeverria, L. Music, D. Music, V. Music,
    Calculation of crystal defects induced in :math:`\mathrm{CaWO_4}` cryogenic detector by 100 eV displacement cascades using a data driven force field,
    Phys. Rev. D 111, 085021 (2025).

Reference data:

* Computed from the DFT total energies provided with the input structures.


Split vacancy
=============

Summary
-------

Performance predicting split vacancy formation energies and relaxed structures for
metal oxides (PBEsol, 531 host compounds, 722 material-cation pairs, 2154 structures)
and stable nitrides (PBE, 144 host compounds, 149 material-cation pairs, 285 structures).

A split vacancy is a stoichiometry-conserving defect complex in which an isolated atomic
vacancy reconstructs into two vacancies and an interstitial
(:math:`V_X \to [V_X + X_i + V_X]`), often with a dramatic energy lowering.

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

where the minima are taken over all initial structures that match the DFT reference
(see metric 3) for a given material-cation pair.

2. Spearman's rank correlation

`Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_
between MLIP and DFT total energies, evaluated as single points on DFT-relaxed NV and
SV structures. A perfect ranking gives a coefficient of 1. The mean across all
material-cation pairs is reported.

3. Match Rate

Fraction of MLIP-relaxed structures that converge to the same geometry as the
DFT-relaxed reference, determined using the pymatgen
`StructureMatcher <https://pymatgen.org/pymatgen.analysis.html#module-pymatgen.analysis.structure_matcher>`_.
The match criterion is a normalised maximum atomic displacement below 0.3 (see
metric 4).

4. Max Dist

Maximum atomic displacement between the MLIP-relaxed and DFT-relaxed matched
structures, normalised by :math:`(V/N)^{1/3}` (wher :math:`V` is the cell volume and
:math:`N` the number of atoms) to give a unitless quantity comparable across
different crystal structures. Only computed for structure pairs that pass the
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
