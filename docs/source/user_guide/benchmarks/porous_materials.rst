================
Porous Materials
================


QMOF
====

Summary
-------

Performance in predicting energies per atom for 13905 metal organic frameworks (MOFs)
from the Quantum MOF Database (QMOF), a public online collection of electronic and
physical properties for more than 20,000 MOFs and coordination polymers[1].


Metrics
-------

MAE

Mean Absolute Error of energy per atom for all systems.

Benchmark speed
------------------

Medium: calculations are expected to take 10s of minutes on a A100 GPU per model.


Data availability
-----------------

Input data:

Structures were taken from the Quantum MOF Database (QMOF), with some structures
removed for compatibility with the MACE_MP paper due to missing elements or
inconsistent DFT.

[1] Rosen A, et al
**Machine learning the quantum-chemical properties of metal–organic frameworks for accelerated materials discovery**
Matter, 2021; 4, 1578-1597, 10.1016/j.matt.2021.02.015

[2] Batatia I, et al
A foundation model for atomistic materials chemistry
J. Chem. Phys. 163, 184110 (2025), https://doi.org/10.1063/5.0297006

Reference data:

* Same as input data
* PBE-D3
