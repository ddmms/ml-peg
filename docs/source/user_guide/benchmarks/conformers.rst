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


TorsionNet500
=============

Summary
-------

Performance in predicting torsion energy barriers for 500 drug-like organic
molecules, each scanned across a systematically sampled dihedral angle.

Metrics
-------

1. Barrier height error

For each fragment, a single point energy is computed for every conformer along
the dihedral scan. The predicted profile is aligned to the reference at its
minimum, and the barrier height is taken as the difference between the maximum
and minimum energy of the profile. The reported metric is the mean absolute
error of the barrier height across all fragments. The overall score is the
mlipaudit per-fragment soft-threshold score on the barrier height error.

A parity plot of the predicted against reference barrier heights is shown on
clicking the metric column.

Computational cost
------------------

Medium: 12,000 single point inference calls on small molecular systems.

Data availability
-----------------

Input structures:

* TorsionNet: A Deep Neural Network to Rapidly Predict Small-Molecule
  Torsional Energy Profiles with the Accuracy of Quantum Mechanics.
  Rai, B. K. et al. J. Chem. Inf. Model. 2022, 62 (4), 785-800.
  DOI: 10.1021/acs.jcim.1c01346

Reference data:

* Recomputed by InstaDeep at the :math:`\omega B97M-D3(BJ)/def2-TZVPPD`
  level of theory (MLIP Audit benchmark suite).
