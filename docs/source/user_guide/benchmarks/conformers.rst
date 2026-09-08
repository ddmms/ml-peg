==========
Conformers
==========

ACONFL
======

Summary
-------

Performance in predicting relative conformer energies of 12 C12H26,
16 C16H34 and 20 C20H42 conformers. Reference data from PNO-LCCSD(T)-F12/ AVQZ calculations.

Metrics
-------

1. Conformer energy error

For each complex, the the relative energy is calculated by taking the difference in energy
between the given conformer and the reference (zero-energy) conformer. This is
compared to the reference conformer energy, calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.

Data availability
-----------------

Input structures:

* Conformational Energy Benchmark for Longer n-Alkane Chains
  Sebastian Ehlert, Stefan Grimme, and Andreas Hansen
  The Journal of Physical Chemistry A 2022 126 (22), 3521-3535
  DOI: 10.1021/acs.jpca.2c02439

Reference data:

* Same as input data
* :math:`PNO-LCCSD(T)-F12/ AVQZ` level of theory: a local, explicitly
  correlated coupled cluster method.


Folmsbee
========

Summary
-------

Performance in predicting relative conformer energies for a set of drug-like
molecules. Each molecule has a small set of conformers (3-10 per molecule)
whose energies are ranked relative to the lowest-energy conformer. Reference
data from DLPNO-CCSD(T) calculations.

Metrics
-------

For each molecule, the relative energy of every conformer is taken with respect
to the lowest-energy reference conformer (set to zero). The predicted energies
are aligned to the same reference conformer and converted to kcal/mol before
being compared against the :math:`DLPNO-CCSD(T)` reference.

1. MAE

The mean absolute error between predicted and reference relative energies,
averaged over all molecules.

2. Conformer Score

For each molecule, the per-molecule MAE and RMSE are passed
through a soft threshold (MAE at 0.5 kcal/mol, RMSE at 1.5 kcal/mol) to give a
value between 0 and 1, and these are averaged across all molecules. Molecules
for which the model fails to produce an energy profile score 0.

Computational cost
------------------

Medium: Minutes on GPU. Minutes to tens of minutes on CPU.

Data availability
-----------------

Input structures:

* Assessing conformer energies using electronic structure and machine learning
  methods
  Dakota Folmsbee, Geoffrey Hutchison
  International Journal of Quantum Chemistry 2020 121 (1) e26381
  DOI: 10.1002/qua.26381

Reference data:

* Same as input data
* :math:`DLPNO-CCSD(T)` level of theory: a local coupled-cluster method.
