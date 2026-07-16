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


HPHT_CH4_H2O
============

Summary
-------

Benchmark tests the capability of MLIPs to reproduce free energy profile of proton hopping reactions
in CH4/H2O liquid from molecular dynamics simulations at 3000 K and 22-69 GPa.

The associated analysis uses a molecular recognition algorithm to collect
reaction coordinate values and compute the unbiased free energy profile associated to the reaction:
H3O + CH4 = H2O + CH5+. The reaction coordinate is d(O–H)− d(C–H), defined as the difference between
the distance from the hydrogen to the nearest oxygen and the distance from the same hydrogen
to the nearestcarbon atom.

Reference profiles were computed at the DFT (GGA) level (PBE+D3).

Metrics
-------

1. Global MAE of predicted profiles with respect to the reference ones.
2. MAE on free energy of reaction (F[product] - F[reactant])
3. MAE on foward free energy barrier (F[transition state] - F[reactant])

Computational cost
------------------

High: Requires to perform MLIP molecular dynamics simulation of 50 ps of a simulation box
containing 488 atoms (52 CH4 and 76 H2O molecules) at 4 distinct pressures.


Data availability
-----------------

Reference data:

T. Thévenet, A. Dian, M. Cioni, D. A. Markovits, D. S. Scandolo, A. France-Lanord, F. Siro Brigiano,
Angew. Chem. Int. Ed. 2026, 65, e20364. https://doi.org/10.1002/anie.202520364
