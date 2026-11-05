===============
Thermochemistry
===============

MSR_ACC_TAE25
=============

Summary
-------

Test the accuracy of predicted total atomization energies (TAEs) for a set of 659 closed-shell, charge neutral and covalently bound molecules with up to 5 heavy atoms.


Metrics
-------

1. Total atomization energy MAE

Accuracy of total atomization energy predictions.

The predicted total atomization energy is calculated as:

$$
\text{TAE} = n\times E(A) - E(M)
$$

where E(M) is the predicted energy of the molecule and E(A) is the predicted energy of each atom at their ground state multiplicity.

The mean absolute error is calculated by comparing the predicted TAE to the reference TAE.


Benchmark speed
------------------

Fast: around 3 minutes per model on CPU.


Data availability
-----------------

Input structures:

* Ehlert, S., Hermann, J., Vogels, T. et al. Accurate Chemistry Collection: Coupled cluster atomization
  energies for broad chemical space. Sci. Data. 2026.

Reference data:

* The reference above provides TAE values at the CCSD(T)/CBS (W1-F12) level.


OSOS_TAE
========

Summary
-------

The OSOS dataset contains 3360 total atomization energies for a diverse set of organic doublet radicals, including both localized and delocalized systems.
The molecular size is limited to 7 heavy atoms.

This benchmark tests the accuracy of MLIP predicted total atomization energies (TAEs) on these open-shell systems.


Metrics
-------

1. Total atomization energy MAE

Accuracy of total atomization energy predictions.

The predicted total atomization energy is calculated as:

$$
\text{TAE} = n\times E(A) - E(M)
$$

where E(M) is the predicted energy of the molecule and E(A) is the predicted energy of each atom at their ground state multiplicity.

The mean absolute error is calculated by comparing the predicted TAE to the reference TAE.


Benchmark speed
------------------

Fast: around 20 minutes per model on CPU.


Data availability
-----------------

Input structures:

* A. Karton. The Open-Shell Organic Systems Database: Diverse CCSD(T)/CBS Reaction Energies
  for Assessing Density Functional Theory and Machine Learning Methods. J. Phys. Chem. A 2026.

Reference data:

* The reference above provides TAE values at the CCSD(T)/CBS (W1-F12) level.
