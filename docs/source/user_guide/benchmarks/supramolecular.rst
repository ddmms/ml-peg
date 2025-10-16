==============
Supramolecular
==============

LNCI16
======

Summary
-------

Performance in predicting host-guest interaction energies for 16 large non-covalent
complexes. These include proteins, DNA, and supramolecular assemblies ranging from 380
up to 1988 atoms with diverse interaction motives. 14 complexes are neutral, and two are
charged with charges +1 (TYK2) and -2 (FXa).

Metrics
-------

Interaction energy error

For each complex, the interaction energy is calculated by taking the difference in energy
between the host-guest complex and the sum of the individual host and guest energies. This is
compared to the reference interaction energy, calculated in the same way.


Computational cost
------------------

low: tests are likely to take minutes to run on CPU.

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
