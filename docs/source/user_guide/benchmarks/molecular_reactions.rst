===================
Molecular Reactions
===================

CRBH20
======

Summary
-------

Performance in predicting reaction energy barriers for the 20 reactions in the CRBH20
dataset. Barriers are computed as the energy difference between the transition state
and the reactant of each reaction.

Metrics
-------

1. MAE

Accuracy of predicted reaction barriers.

For each of the 20 reactions, the barrier is calculated from single point energies of
the reactant and transition state structures. The mean absolute error against the
reference barriers is reported in kcal/mol.

Computational cost
------------------

Low: tests involve single point calculations on 40 small molecular structures, and are
likely to take less than a minute to run on CPU.

Data availability
-----------------

Input structures:

* Appendix B.5 of: Batatia, I. et al. A foundation model for atomistic materials
  chemistry. arXiv:2401.00096. https://doi.org/10.48550/arXiv.2401.00096

Reference data:

* Same as input data
* DFT (r2SCAN)


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

For each pair the reaction energy is the energy difference between the two
tautomers. The predicted reaction energies are compared against the
:math:`\omega B97M-D3(BJ)/def2-TZVPPD` reference reaction energies.

1. MAE

The mean absolute error between predicted and reference reaction energies,
in kcal/mol, averaged over all pairs. Pairs on which inference fails are
excluded from the average. Reported for information only (not scored).

2. Tautomer Score

For each pair the absolute deviation between the predicted
and reference reaction energy is passed through a soft threshold (at 0.05
kcal/mol) to give a value between 0 and 1, and these are averaged across all
pairs. Pairs for which the model fails to produce an energy score 0.

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
