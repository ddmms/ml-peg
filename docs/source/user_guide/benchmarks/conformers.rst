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

Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.

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

TorsionNet500CCSDT
===================

Summary
-------

Performance in predicting torsional energy profiles for 500 diverse organic
molecular fragments. Reference data from DLPNO-CCSD(T)/CBS calculations.

Metrics
-------

1. RMSE of relative torsional energy profile
2. MAE of relative torsional energy profile

For each fragment, a torsion scan samples the energy at a series of dihedral
angles. Both the reference and predicted energies are mean-centered per scan,
so only the shape of the profile is compared rather than absolute energy
offsets. RMSE and MAE are calculated between the reference and predicted
profiles for each scan, then averaged across all 500 fragments.

Computational cost
------------------

Medium: tests are likely to take minutes to run on GPU, or less than an hour on
CPU for each model, since each fragment requires a single-point energy
calculation across ~20 conformers in its torsion scan, repeated for all 500
fragments.

Data availability
-----------------

Input structures:

* B. K. Rai, V. Sresht, Q. Yang, R. Unwalla, M. Tu, A. M. Mathiowetz, and
  G. A. Bakken, TorsionNet: A Deep Neural Network to Rapidly Predict
  Small-Molecule Torsional Energy Profiles with the Accuracy of Quantum
  Mechanics, Journal of Chemical Information and Modeling 62 (2022), 785-800.
  PMID: 35119861.

Reference data:

* J. L. Weber, R. D. Guha, G. Agarwal, Y. Wei, A. A. Fike, X. Xie,
  J. Stevenson, B. Santra, R. A. Friesner, K. Leswing, M. D. Halls, R. Abel,
  and L. D. Jacobson, Efficient Long-Range Machine Learning Force Fields for
  Liquid and Materials Properties, arXiv:2505.06462 (2025).
* DLPNO-CCSD(T)/CBS level of theory.
