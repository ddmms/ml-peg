==========================
Transition Metal Complexes
==========================

3dTMV
=======

Summary
-------

Performance in predicting vertical ionization energies for 28 transition metal
complexes.

Metrics
-------

1. Ionization energy error

For each complex, the ionization energy is calculated by taking the difference in energy
between the complex in its oxidized state and initial state, which differ by one electron
and spin multiplicity. This is compared to the reference ionization energy, calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.

Data availability
-----------------

Input structures:

* Toward Benchmark-Quality Ab Initio Predictions for 3d Transition Metal Electrocatalysts: A Comparison of CCSD(T) and ph-AFQMC
Hagen Neugebauer, Hung T. Vuong, John L. Weber, Richard A. Friesner, James Shee, and Andreas Hansen
Journal of Chemical Theory and Computation 2023 19 (18), 6208-6225
DOI: 10.1021/acs.jctc.3c00617

Reference data:

* Same as input data
* ph-AFQMC level of theory: Auxiliary-Field Quantum Monte Carlo.
