=================
Molecular Systems
=================

GMTKN55
=======

Summary
-------

Performance in evaluating gas-phase chemical accuracy for main group thermochemistry,
kinetics and noncovalent interactions, through 55 test sets, totalling 1,505 relative
energies, categorised into five chemical domains.


Metrics
-------

1. Small systems

Weighted mean absolute deviation (MAD) of basic properties, such as atomic energies,
ionisation potentials, and electron affinities, of small systems.

For each system, the relative energy is calculated and compared to the reference
energy. The MAD is calculated for each of the subsets within this category, including
only neutral singlet systems. A weighted sum is calculated by multiplying each subset
error by a weight and the number of systems in the subset. This is divided by the total
number of systems within these subsets.

2. Large systems

Weighted mean absolute deviation (MAD) of reaction energies of large systems and
isomerisation energies.

Same as (1), for the appropriately categorised subsets.

3. Barrier heights

Weighted mean absolute deviation (MAD) of reaction barrier heights for transition state
energetics for fundamental organic reactions.

Same as (1), for the appropriately categorised subsets.

4. Intramolecular NCIs

Weighted mean absolute deviation (MAD) of intramolecular noncovalent interactions for
conformational energetics and hydrogen bonding.

Same as (1), for the appropriately categorised subsets.

5. Intermolecular NCIs

Weighted mean absolute deviation (MAD) of intermolecular noncovalent interactions for
dimers, clusters, and host-guest complexes.


5. All (WTMAD)

Weighted mean absolute deviation (MAD) of all subsets.

Same as (1), for all subsets.


Computational cost
------------------

Low: tests are likely to take minutes to run on CPU.


Data availability
-----------------

Input structures:

* L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich, A. Najibi, and S. Grimme, A look at the
  density functional theory zoo with the advanced gmtkn55 database for general main
  group thermochemistry, kinetics and noncovalent interactions, Physical Chemistry
  Chemical Physics 19, 32184 (2017).

Reference data:

* Same as input data
* CCSD(T)


Wiggle150
=========

Summary
-------

Performance in predicting relative energies between 150 strained conformers of
adenosine, benzylpenicillin, and efavirenz molecules (50 each) and their geometry optimised structures.


Metrics
-------

1. Relative energy MAE

Accuracy of relative energy predictions.

For each molecule, 50 relative energies are calculated by comparing the predicted energy of the
DLPNO-CCSD(T)/CBS geometry optimised structure to the energies of its 50 strained conformers. The
mean absolute error is reported over all 150 conformers.


Computational cost
------------------

Low: tests are likely to take less than a minute to run on CPU.


Data availability
-----------------

Input structures:

* Brew, R. R. et al. Wiggle150: Benchmarking Density Functionals and Neural Network Potentials
  on Highly Strained Conformers. J. Chem. Theory Comput. 21, 3922-3929 (2025).


Reference data:

* Same as input data
* DLPNO-CCSD(T)/CBS


BMIM Cl RDF
===========

Summary
-------

Tests whether MLIPs incorrectly predict covalent bond formation between chloride
anions (Cl⁻) and carbon atoms in 1-butyl-3-methylimidazolium (BMIM⁺) cations.
Such Cl-C bonds should NOT form in the ionic liquid under normal conditions.

This benchmark runs NVT molecular dynamics simulations of BMIM Cl at
353.15 K and analyses the Cl-C RDF to detect any unphysical bond formation.


Metrics
-------

1. Cl-C Bonds Formed

Binary metric indicating whether unphysical Cl-C bonds formed during the MD simulation.

The Cl-C RDF is computed from the MD trajectory. If the RDF shows a peak (g(r) > 0.1)
at distances below 2.5 Å, this indicates bond formation and the model fails the test.

* 0 = no bonds formed (correct physical behaviour)
* 1 = bonds formed (unphysical, model failure)


Computational cost
------------------

Medium: tests require running 10,000 steps of Langevin MD for a system of 10 ion
pairs, which may take tens of minutes on GPU.


Data availability
-----------------

Input structures:

* Generated using molify from SMILES representations of BMIM⁺ (CCCCN1C=C[N+](=C1)C)
  and Cl⁻ ions, packed to experimental density of 1052 kg/m³ at 353.15 K.
* Zills, F. molify: Molecular Structure Interface. Journal of Open Source Software
  10, 8829 (2025). https://doi.org/10.21105/joss.08829
* Density from: Yang, F., Wang, D., Wang, X. & Liu, Z. Volumetric Properties of
  Binary and Ternary Mixtures of Bis(2-hydroxyethyl)ammonium Acetate with Methanol,
  N,N-Dimethylformamide, and Water at Several Temperatures. J. Chem. Eng. Data 62,
  3958-3966 (2017). https://doi.org/10.1021/acs.jced.7b00654



BDEs
====

Summary
-------

Performance in predicting C-H bond dissociation energies (BDEs) for 60 CYP3A4 drug-like
substrates (CHO elements only), comprising 1117 sp3 C-H bonds across all molecules.


Metrics
-------

1. Direct BDE

Mean absolute error (MAE) of predicted BDEs against DFT reference values, in kcal/mol.

For each molecule, BDEs are computed as: BDE = E(radical) + E(H) − E(molecule), where
energies are evaluated on DFT-optimised geometries.

2. BDE rank

Mean Kendall's τ rank correlation of predicted BDE rankings against reference rankings,
evaluated per molecule and averaged across all 60 compounds.

3. Direct BDE (MLFF opt)

Same as (1), but geometries are first relaxed using the MLFF before evaluating energies.

4. BDE rank (MLFF opt)

Same as (2), but using MLFF-optimised geometries.


Computational cost
------------------

Medium: the DFT geometry tests are fast, but the MLFF geometry optimisation tests may
take several minutes per model on CPU.


Data availability
-----------------

Input structures:

* Gelzinyte, E. et al. Transferable Machine Learning Interatomic Potential for Bond
  Dissociation Energy Estimation. J. Chem. Theory Comput. 20, 164-177 (2024).
  DOI: 10.1021/acs.jctc.3c00710

Reference data:

* Same as input data
* B3LYP-D3BJ/def2-SV(P)
