==============
Supramolecular
==============

LNCI16
=======

Summary
-------

Performance in predicting host-guest interaction energies for 16 large non-covalent
complexes. These include proteins, DNA, and supramolecular assemblies ranging from 380
up to 1988 atoms with diverse interaction motives. 14 complexes are neutral, and two are
charged with charges +1 (TYK2) and -2 (FXa).

Metrics
-------

1. Interaction energy error

For each complex, the interaction energy is calculated by taking the difference in energy
between the host-guest complex and the sum of the individual host and guest energies. This is
compared to the reference interaction energy, calculated in the same way.

Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.

Data availability
-----------------

Input structures:

* J. Gorges, B. Bädorf, A. Hansen, and S. Grimme, ‘LNCI16 - Efficient Computation of
  the Interaction Energies of Very Large Non-covalently Bound Complexes’, Synlett, vol.
  34, no. 10, pp. 1135–1146, Jun. 2023, doi: 10.1055/s-0042-1753141.

* Associated GitHub repository: https://github.com/grimme-lab/benchmark-LNCI16

.. note::

    As described in the above repository, the originally released dataset contained
    some incorrect values. We have used the corrected values for this benchmark.

Reference data:

* Same as input data
* :math:`{\omega}\text{B97X-3c}` level of theory: a composite range-separated hybrid DFT
  method with a refitted D4 dispersion correction.


S30L
====

Summary
-------

Performance in predicting supramolecular binding energies for the S30L set of 30 large
host–guest complexes. Systems contain up to ~200 atoms and feature a wide range of
interaction motifs, including hydrogen and halogen bonding, π–π stacking, CH–π
contacts, nonpolar dispersion, and cation–dipolar interactions. Net charges span −1 to
+4, where 8 of the 30 complexes are charged. These ΔE_emp values serve as the benchmark
dataset for assessing quantum-chemical methods on large noncovalent complexes.


Metrics
-------

1. Total binding energy error

For each complex, the binding energy is calculated by taking the difference in energy
between the host-guest complex and the sum of the individual host and guest energies. This is
compared to the reference binding energy, calculated in the same way.

2. Charged binding energy error

For each charged complex, the binding energy is calculated by taking the difference in energy
between the host-guest complex and the sum of the individual host and guest energies. This is
compared to the reference binding energy, calculated in the same way.

3. Neutral binding energy error

For each neutral complex, the binding energy is calculated by taking the difference in energy
between the host-guest complex and the sum of the individual host and guest energies. This is
compared to the reference binding energy, calculated in the same way.


Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.


Data availability
-----------------

* R. Sure and S. Grimme, ‘S30L - Comprehensive Benchmark of Association (Free) Energies
  of Realistic Host–Guest Complexes’, J. Chem. Theory Comput., vol. 11, no. 8, pp. 3785–3801, Aug. 2015, doi: 10.1021/acs.jctc.5b00296.
* Stuctures download found in SI
    * TPSS-D3/def2-TZVP geometries of all complexes as Cartesian coordinates.

Reference data:

* The Supporting Information also provides the benchmark binding energies.
    * These are empirical gas-phase reference values (ΔE_emp), obtained by
      back-correcting the experimental association free energies with theoretical
      estimates of vibrational and solvation contributions.


PLF547
======

Summary
-------

Performance in predicting intramolecular hydrogen bonding and side–chain/backbone
contacts in polypeptide fragments for the PLF547 set of 547 shorter peptide-like
fragments.


Metrics
-------

1. MAE

For each complex, the interaction energy is calculated by taking the difference in
energy between the ligand-fragment complex and the sum of the individual ligand and
protein fragment energies. This is compared to the reference interaction energy.


Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.


Data availability
-----------------

Input structures:

* K. Kříž and J. Řezáč, "Benchmarking of Semiempirical Quantum-Mechanical Methods on
  Systems Relevant to Computer-Aided Drug Design", Journal of Chemical Information and
  Modeling, vol. 60, no. 3, pp. 1453-1460, 2020, doi: 10.1021/acs.jcim.9b01171

  * DLPNO-CCSD(T)/CBS

Reference data:

Same as input structures


PLA15
=====

Summary
-------

Performance in predicting protein–ligand active-site interaction energies for the
PLA15 set of 15 complexes. Systems range from 259 to 584 atoms and contain complete
active sites. Ligands contain 37–95 atoms with net charges of −1, 0, or +1 and all contain
aromatic heterocycles. Five ligands contain either divalent of tetrahedral sulfur atoms, and
four and three of them contain F and Cl atoms, respectively.

Metrics
-------

1. Total MAE

For each complex, the interaction energy is calculated by taking the difference in
energy between the protein-ligand complex and the sum of the individual protein and
ligand energies. The MAE is computed by comparing predicted interaction energies to
reference interaction energies across all 15 systems.

2. Pearson's r²

The squared Pearson correlation coefficient between predicted and reference interaction
energies, measuring the proportion of variance in the reference values explained by the
model predictions.

3. Ion-Ion MAE

For each complex where both protein and ligand fragments have non-zero charges, the
interaction energy error is calculated. This metric reports the MAE for these ion-ion
interaction systems.

4. Ion-Neutral MAE

For each complex where one fragment (protein or ligand) has a non-zero charge and the
other is neutral, the interaction energy error is calculated. This metric reports the
MAE for these ion-neutral interaction systems.


Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.

Data availability
-----------------

Input structures:

* K. Kříž and J. Řezáč, ‘protein ligand - Benchmarking of Semiempirical
  Quantum-Mechanical Methods on Systems Relevant to Computer-Aided Drug Design’, J.
  Chem. Inf. Model., vol. 60, no. 3, pp. 1453–1460, Mar. 2020, doi:
  10.1021/acs.jcim.9b01171.
* Structures download found in SI

Reference data:

* The Supporting Information also provides the interaction energies.
    * The benchmark interaction energies are based on a combination of explicitly
      correlated MP2-F12 calculations and a DLPNO-CCSD(T) correction
