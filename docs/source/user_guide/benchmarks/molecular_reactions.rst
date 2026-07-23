===================
Molecular reactions
===================

Tautomers
=========

Summary
-------

Performance in predicting the relative energy of tautomer pairs. Each system is
a pair of tautomers (constitutional isomers differing in the position of a
proton and an associated double bond), and the benchmark measures how well a
model reproduces the energy difference between the two forms. The structures are
taken from the Tautobase dataset and are pre-optimised; only single-point
energies are evaluated.

Metrics
-------

1. Reaction energy MAE

For each pair the reaction energy is the energy difference between the two
tautomers. The mean absolute error (MAE) between the predicted and reference
reaction energies is reported in kcal/mol, averaged across all pairs. Pairs on
which inference fails are excluded from the average.

Computational cost
------------------

Low: only single-point energies are evaluated, so tests run quickly even for the
full dataset. Minutes on CPU and GPU.

Data availability
-----------------

Input structures:

* Tautobase: an open tautomer database.
  Wahl, O.; Sander, T. *J. Chem. Inf. Model.* 2020, 60 (3), 1085-1089.
  DOI: 10.1021/acs.jcim.0c00035

Reference data:

* Same as input data
