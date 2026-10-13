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

Benchmark speed
------------------

Fast: tests are likely to take minutes to run on CPU.

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

Benchmark speed
------------------

Fast: minutes on GPU, and minutes to tens of minutes on CPU. The benchmark is
marked ``fast``.

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


Sellers_Tors
============

Summary
-------

Performance in predicting torsional energy profiles of 62 neutral drug-like
fragments (alkyl, aryl, biaryl and aryl amide torsions). Each fragment is scanned
over 36 dihedral angles in 10° steps, giving 2232 conformers. Reference data from
CCSD(T)/CBS single-point calculations on MP2/6-311+G**-optimised constrained geometries.

Metrics
-------

For each torsion scan, the predicted relative energy of every scan point is taken
with respect to the scan point that is lowest in energy at the reference level of
theory, and converted to kcal/mol. The reference relative energies are calculated
in the same way.

1. MAE

The mean absolute error between predicted and reference relative energies,
over all scan points.

2. RMSE

The root mean squared error between predicted and reference relative energies,
over all scan points.

3. Barrier MAE

For each scan, the torsional barrier is the difference between the maximum and
minimum relative energies. The mean absolute error between predicted and
reference barriers is averaged over all fragments.

Benchmark speed
------------------

Fast: minutes on GPU, and minutes to tens of minutes on CPU. The benchmark is
marked ``fast``. It took around 13 minutes for ``mace-omol`` on an Apple Silicon
CPU (2232 single points).

Data availability
-----------------

Input structures:

* A Comparison of Quantum and Molecular Mechanical Methods to Estimate Strain
  Energy in Druglike Fragments
  Benjamin D. Sellers, Natalie C. James, Alberto Gobbi
  Journal of Chemical Information and Modeling 2017 57 (6), 1265-1275
  DOI: 10.1021/acs.jcim.6b00614

Reference data:

* Same as input data
* :math:`CCSD(T)/CBS` level of theory, on MP2/6-311+G**-optimised constrained geometries.
