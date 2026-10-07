================
Porous Materials
================


QMOF
====

Summary
-------

Quantum MOF Database, a public online collection of electronic and physical properties for more than 20,000 metal–organic frameworks (MOFs) and coordination polymers[1].
This benchamrk uses only the subset which was compatible with the MACE_MP paper. Removed structures involve missing elements or inconsistent DFT, 13912 [2].
The DFT is computed at PBE level.


Metrics
-------

MAE

Mean Absolute Error of energy per atom for all systems

Benchmark speed
------------------

Medium: shall tale 10s of minutes on a A100 GPU


Reference data

[1] Rosen A, et al
**Machine learning the quantum-chemical properties of metal–organic frameworks for accelerated materials discovery**
Matter, 2021; 4, 1578-1597, 10.1016/j.matt.2021.02.015
[2] Batatia I, et al
A foundation model for atomistic materials chemistry
J. Chem. Phys. 163, 184110 (2025), https://doi.org/10.1063/5.0297006
