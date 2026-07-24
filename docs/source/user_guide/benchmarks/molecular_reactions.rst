===================
Molecular reactions
===================

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
