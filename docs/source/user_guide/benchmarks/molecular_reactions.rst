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


Grambow barrier heights
=======================

Summary
-------

Performance in predicting reaction barrier heights for elementary organic
reactions from the Grambow dataset, comprising almost 12,000 reactant, product
and transition state triplets of small neutral molecules containing H, C, N and
O. The benchmark is targeted at barrier heights and reaction energies.

Metrics
-------

1. Barrier height MAE

For each reaction, a single point energy is computed for the reactant, product
and transition state. The activation energy (barrier height) is the transition
state energy minus the reactant energy, and the reaction energy is
the product energy minus the reactant energy. The reported metrics are the mean
absolute errors of the activation energy and the reaction energy against the
reference.

2. Grambow score

The overall score is the mlipaudit soft-threshold score combining the
activation energy and reaction energy errors.

A density scatter plot shows the predicted against reference activation energies
on clicking the barrier height column.

Computational cost
------------------

Medium: around 36,000 single point inference calls (three states per reaction). Minutes on CPU, tens of minutes to hours on GPU.

Data availability
-----------------

Input structures:

* Grambow, C.A., Pattanaik, L. & Green, W.H. Reactants, products, and
  transition states of elementary chemical reactions based on quantum
  chemistry. Sci Data 7, 137 (2020). DOI: 10.1038/s41597-020-0460-4

Reference data:

* Same as input data
* :math:`\omega B97X-D3/def2-TZVP` level of theory.
