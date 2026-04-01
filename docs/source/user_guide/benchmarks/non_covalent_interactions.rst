=========================
Non Covalent Interactions
=========================

IONPI19
=======

Summary
-------

Performance in predicting interaction energies for 19 complexes, each consisting of
an ion and an aromatic pi-electron system. The ions include lithium, sodium, potassium,
cyclopropenyl cation, chloride, nitrate, thiocyanate. The dataset also includes
relative conformational energy tests for aromatic ions involving pi--ion interactions.

Metrics
-------

1. Interaction energy error

For each complex, the interaction energy is calculated by taking the difference in energy
between the complex and the sum of the individual fragment energies. This is
compared to the reference interaction energy, calculated in the same way. For examples 18 and 19,
the energy is computed by taking the difference between the two conformers.

Computational cost
------------------

Low: tests are likely to take seconds to run on GPU.

Data availability
-----------------

Input structures:

* Spicher, S., Caldeweyher, E., Hansen, A. and Grimme, S., 2021.
  Benchmarking London dispersion corrected density functional theory for
  noncovalent ion–π interactions. Physical Chemistry Chemical Physics, 23(20),
  pp.11635-11648.

Reference data:

* DLPNO-CCSD(T)/CBS


NCIA D1200
==========

Summary
-------

Performance in predicting London dispersion interactions across diverse molecular
dimers. The NCIA D1200 dataset contains 1200 different dimers.

Metrics
-------

1. Interaction energy error

For each dimer, the interaction energy is computed by subtracting the sum of the monomer
energies from the dimer energy. This is compared to the reference interaction energy,
calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* Řezáč, J., 2022. Non-Covalent Interactions Atlas benchmark data sets 5:
  London dispersion in an extended chemical space. Physical Chemistry
  Chemical Physics, 24(24), pp.14780-14793.

Reference data:

* CCSD(T)/CBS


NCIA D442x10
============

Summary
-------

Performance in predicting London dispersion interactions across diverse molecular
dimers. The NCIA D442x10 dataset contains 442 10-point dissociation curves, each
covering a range from 0.8 to 2.0 times the equilibrium intermolecular separation.

Metrics
-------

1. Interaction energy error

For each dimer, the interaction energy is computed by subtracting the sum of the monomer
energies from the dimer energy. This is compared to the reference interaction energy,
calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* Řezáč, J., 2022. Non-Covalent Interactions Atlas benchmark data sets 5:
  London dispersion in an extended chemical space. Physical Chemistry
  Chemical Physics, 24(24), pp.14780-14793.

Reference data:

* CCSD(T)/CBS


NCIA HB375x10
=============

Summary
-------

Performance in predicting hydrogen bonding interactions across diverse molecular
dimers. The NCIA HB375x10 dataset contains 375 10-point dissociation curves, each
covering a range from 0.8 to 2.0 times the equilibrium intermolecular separation.

Metrics
-------

1. Interaction energy error

For each dimer, the interaction energy is computed by subtracting the sum of the monomer
energies from the dimer energy. This is compared to the reference interaction energy,
calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* Rezac, J., 2020. Non-covalent interactions atlas benchmark data sets: Hydrogen bonding.
  Journal of chemical theory and computation, 16(4), pp.2355-2368.

Reference data:

* CCSD(T)/CBS


NCIA IHB100x10
==============

Summary
-------

Performance in predicting hydrogen bonding interactions across diverse molecular
dimers, including one neutral molecule and one molecular ion. The NCIA IHB100x10 dataset
contains 100 10-point dissociation curves, each covering a range from 0.8 to 2.0 times
the equilibrium intermolecular separation.

Metrics
-------

1. Interaction energy error

For each dimer, the interaction energy is computed by subtracting the sum of the monomer
energies from the dimer energy. This is compared to the reference interaction energy,
calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* Rezac, J., 2020. Non-covalent interactions atlas benchmark data sets: Hydrogen bonding.
  Journal of chemical theory and computation, 16(4), pp.2355-2368.

Reference data:

* CCSD(T)/CBS


NCIA HB300SPXx10
================

Summary
-------

Performance in predicting hydrogen bonding interactions across diverse molecular
dimers, containing sulfur, phosphorus, and halogens as the H-bond accepting atoms.
The NCIA IHB100x10 dataset contains 300 10-point dissociation curves, each covering
a range from 0.8 to 2.0 times the equilibrium intermolecular separation.

Metrics
-------

1. Interaction energy error

For each dimer, the interaction energy is computed by subtracting the sum of the monomer
energies from the dimer energy. This is compared to the reference interaction energy,
calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* Řezáč, J., 2020. Non-covalent interactions atlas benchmark data sets 2:
  Hydrogen bonding in an extended chemical space.

Reference data:

* CCSD(T)/CBS


NCIA SH250x10
=============

Summary
-------

Performance in predicting sigma-hole interaction energies across diverse molecular dimers.
The NCIA SH250x10 dataset contains 250 10-point dissociation curves, each covering
a range from 0.8 to 2.0 times the equilibrium intermolecular separation.

Metrics
-------

1. Interaction energy error

For each dimer, the interaction energy is computed by subtracting the sum of the monomer
energies from the dimer energy. This is compared to the reference interaction energy,
calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* Kříž, K. and Řezáč, J., 2022. Non-covalent interactions atlas benchmark data sets 4:
  σ-hole interactions. Physical Chemistry Chemical Physics, 24(24), pp.14794-14804.

Reference data:

* CCSD(T)/CBS


NCIA R739x5
===========

Summary
-------

Performance in predicting close-contact interaction energies across diverse molecular dimers.
The NCIA R739x5 dataset contains 739 5-point curves.

Metrics
-------

1. Interaction energy error

For each dimer, the interaction energy is computed by subtracting the sum of the monomer
energies from the dimer energy. This is compared to the reference interaction energy,
calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* Kriz, K., Novacek, M. and Rezac, J., 2021. Non-covalent interactions atlas benchmark data sets 3:
  Repulsive contacts. Journal of Chemical Theory and Computation, 17(3), pp.1548-1561.

Reference data:

* CCSD(T)/CBS
