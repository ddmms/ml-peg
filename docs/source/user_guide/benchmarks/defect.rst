======
Defect
======

Defectstab
==========

Summary
-------

This benchmark evaluates formation energies across four distinct subsets covering
different material systems: iron self-interstitial atoms (SIA), boron carbide
stoichiometry, boron carbide point defects, and a methylammonium lead iodide
vacancy. The overall RMSD metric is the average of the per-subset RMSDs.
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

RMSD

Root Mean Square Deviation of formation energies compared to DFT data.
The RMSD is computed independently for each subset, and the overall score
reported in the table is the average of the per-subset RMSDs.
Per-subset values are reported as separate columns in the app table.

For Defectstab, the ``bad`` threshold for each subset RMSD is scaled from the
reference formation-energy magnitude of that subset:

.. math::
  \mathrm{bad} = 0.25 \times P90\left(|E_\mathrm{ref}|\right)

where :math:`P90` is the 90th percentile of the absolute reference formation
energies :math:`|E_\mathrm{ref}|` in that subset. In other words, it is the
value below which 90% of the subset reference magnitudes lie. This gives a
robust high-energy scale for each subset while avoiding over-sensitivity to a
single extreme point. The total RMSD threshold is computed in the same way
using all Defectstab reference values together.

Computational cost
------------------

Low: The geometries are static, requiring only single-point energy calculations
for the configurations and reference structures.

Data availability
-----------------

Input structures:

* Subset ``fe`` of the ``Defectstab`` dataset (:math:`\mathrm{Fe}` SIA configurations).

  * A. Allera, T.D. Swinburne, A.M. Goryaeva, B. Bienvenu, F. Ribeiro, M. Perez, M.-C. Marinica, D. Rodney,
    Activation entropy of dislocation glide in body-centered cubic metals from atomistic simulations,
    Nat Commun 16, 8367 (2025).

  * A.M. Goryaeva, C. Lapointe, C. Dai, J. Dérès, J.-B. Maillet, M.-C. Marinica,
    Reinforcing materials modelling by encoding the structures of defects in crystalline solids into distortion scores,
    Nat Commun 11, 4691 (2020).

* Subset ``boroncarbide_stoichiometry`` and ``boroncarbide_defects`` of the ``Defectstab`` dataset (Boron Carbide structures).

  * G. Roma, K. Gillet, A. Jay, N. Vast, G. Gutierrez,
    Understanding first-order Raman spectra of boron carbides across the homogeneity range,
    Phys. Rev. Materials 5, 063601 (2021).

* Subset ``mapi_tetragonal`` of the ``Defectstab`` dataset (MAPI tetragonal structures).

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

Metrics
-------

1. GlobalMin (Weight: 0.5)

Correct identification of the Global Minimum.
For each subset, this metric checks if the configuration predicted to have the
lowest energy matches the reference global minimum.
It is a binary score (1.0 for a match, 0.0 otherwise) averaged over all
evaluated subsets.
This metric assesses the model's ability to find the most stable structure
within the provided set of configurations.

2. Top5_Spearman (Weight: 1.0)

Spearman rank correlation coefficient calculated on the
5 highest energy (least stable) configurations.
For each subset, we identify the 5 configurations with the highest reference
energies (the least stable ones) and compute the Spearman correlation between
their reference and predicted rankings.
This metric focuses on the model's ability to correctly order high-energy states,
which are often critical for understanding transition pathways or high-temperature
behavior.
The final score is the average correlation across all evaluated subsets.

Per-subset values for both metrics are reported as separate columns in the app table.

Computational cost
------------------

Low: Requires single-point energy calculations for the configurations in each
subset.

Data availability
-----------------

Input structures:

* Subset ``fe`` of the ``Relastab`` dataset (:math:`\mathrm{Fe}` SIA configurations).

  * A. Allera, T.D. Swinburne, A.M. Goryaeva, B. Bienvenu, F. Ribeiro, M. Perez, M.-C. Marinica, D. Rodney,
    Activation entropy of dislocation glide in body-centered cubic metals from atomistic simulations,
    Nat Commun 16, 8367 (2025).

  * A.M. Goryaeva, C. Lapointe, C. Dai, J. Dérès, J.-B. Maillet, M.-C. Marinica,
    Reinforcing materials modelling by encoding the structures of defects in crystalline solids into distortion scores,
    Nat Commun 11, 4691 (2020).

* Subset ``cawo4`` of the ``Relastab`` dataset (:math:`\mathrm{CaWO_4}` interstitial configurations).

  * G. Soum-Sidikov, A. Boisard, D. L010, M. Loidl, O. Stézowski, A. Music, G. Music, Y. Zeng, R. Cong, T. Yang, A. Echeverria, L. Music, D. Music, V. Music,
    Calculation of crystal defects induced in :math:`\mathrm{CaWO_4}` cryogenic detector by 100 eV displacement cascades using a data driven force field,
    Phys. Rev. D 111, 085021 (2025).

Reference data:

* Computed from the DFT total energies provided with the input structures.
