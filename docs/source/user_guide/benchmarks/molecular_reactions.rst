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
