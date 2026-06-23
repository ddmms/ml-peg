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

1. Conformer energy error

For each molecule, the relative energy of every conformer is taken with
respect
to the lowest-energy reference conformer (set to zero). The predicted energies
are aligned to the same reference conformer and converted to kcal/mol. The
mean
absolute error (MAE) between predicted and reference relative energies is
reported, averaged across all systems.

Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.

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
