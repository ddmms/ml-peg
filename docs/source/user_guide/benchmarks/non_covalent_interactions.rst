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

Benchmark speed
------------------

Fast: tests are likely to take seconds to run on GPU.

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

Benchmark speed
------------------

Fast: tests are likely to take minutes to run on GPU.

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

Benchmark speed
------------------

Fast: tests are likely to take minutes to run on GPU.

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

Benchmark speed
------------------

Fast: tests are likely to take minutes to run on GPU.

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

Benchmark speed
------------------

Fast: tests are likely to take minutes to run on GPU.

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

Benchmark speed
------------------

Fast: tests are likely to take minutes to run on GPU.

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

Benchmark speed
------------------

Fast: tests are likely to take minutes to run on GPU.

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

Benchmark speed
------------------

Fast: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* Kriz, K., Novacek, M. and Rezac, J., 2021. Non-covalent interactions atlas benchmark data sets 3:
  Repulsive contacts. Journal of Chemical Theory and Computation, 17(3), pp.1548-1561.

Reference data:

* CCSD(T)/CBS

NABP30
======

Summary
-------

Performance in predicting interaction and binding energies of 30 hydrogen-bonded nucleic
acid base pairs (NABP30), ranging from the weak, nonpolar adenine...difluorotoluene pair to the protonated
cytosine...cytosine+ pair. The set covers Watson-Crick, wobble, purine-purine and
pyrimidine-pyrimidine pairs, including thio-, oxo- and amino-modified bases and planar and
nonplanar variants of some pairs.

Metrics
-------

1. Interaction energy error

For each base pair, the energies of the base pair and of its two bases are calculated at
the reference geometry (RI-MP2 optimised), with each base kept as it is in the base pair.
The interaction energy is the energy of the base pair minus the energies of the two bases.
Nothing is relaxed, so it does not include the energy needed to deform the bases on
binding.

This is compared to the reference RI-MP2/CBS interaction energy: the reference binding
energy described below minus the reference deformation energy of the bases.

2. Binding energy error

For each base pair, the base pair and the two isolated bases are relaxed with the model
(BFGS, maximum force 0.002 eV/Å). Each of the 13 distinct bases is relaxed once, starting
from its geometry in a base pair, and used for every base pair that contains it. As amino
groups are planar in many base pairs but pyramidal in the isolated bases, each base is
relaxed from three starting structures, randomly displaced by 0.05 Å, and the lowest energy
is used. Base pairs that were optimised with Cs symmetry in the reference work are kept
planar; the other base pairs are randomly displaced by 0.01 Å, so that they can leave a
planar starting structure. The binding energy is the energy of the relaxed base pair minus
the energies of the two relaxed bases, so it includes the energy needed to deform the bases
on binding. A base pair is counted as failed if a relaxation does not converge or changes
covalent bonds.

This is compared to the reference RI-MP2/CBS binding energy (aug-cc-pVDZ -> aug-cc-pVTZ,
Helgaker extrapolation, counterpoise-corrected, including the monomer deformation energy).

Thresholds
----------

Both metrics use the thresholds good = 0.5 kcal/mol and bad = 2.0 kcal/mol.

* Good: the reference work states that an accuracy of 0.5 kcal/mol for hydrogen-bonded
  base pairs requires an extrapolation from larger basis sets and a CCSD(T) correction,
  which are not available for all 30 base pairs. Differences from the reference below
  0.5 kcal/mol are therefore within its accuracy, and a smaller error does not score
  higher.
* Bad: 2.0 kcal/mol is in the range of the errors of the medium-quality methods that the
  reference work compares with the same reference energies. From its Table 3, the mean
  absolute errors are 1.4 kcal/mol for the Cornell et al. (AMBER) force field (17 base
  pairs), 2.4 kcal/mol for B3LYP/6-31G** (27 base pairs) and 2.8 kcal/mol for
  MP2/6-31G*(0.25)//HF/6-31G** (23 base pairs). A model with an error of 2.0 kcal/mol is
  not an improvement on these methods, and scores 0, in particular as it is worse than the cheap AMBER force field.
* The same thresholds are used for both metrics, as both are compared with the same
  reference energies: the reference interaction energy is the reference binding energy
  minus the deformation energy of the bases.

Benchmark speed
---------------

Fast: tests are likely to take minutes to run on GPU (about 5 minutes for mace-mp-0a on
one NVIDIA H200).

Data availability
-----------------

Input structures:

* Sponer, J., Jurecka, P. and Hobza, P., 2004. Accurate interaction energies of
  hydrogen-bonded nucleic acid base pairs. Journal of the American Chemical Society,
  126(32), pp.10142-10151.
* Adenine...adenine 3 (not in the supporting information of the above): Jurecka, P.,
  Sponer, J., Cerny, J. and Hobza, P., 2006. Benchmark database of accurate (MP2 and
  CCSD(T) complete basis set limit) interaction energies of small model complexes, DNA
  base pairs, and amino acid pairs. Physical Chemistry Chemical Physics, 8(17),
  pp.1985-1993.

Reference data:

* Sponer, J., Jurecka, P. and Hobza, P., 2004 (as above)
* MP2/CBS
