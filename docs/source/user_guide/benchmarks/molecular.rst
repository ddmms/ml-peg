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
* PBE-D3(BJ)
