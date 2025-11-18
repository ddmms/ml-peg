============
Bulk Crystal
============

Lattice constants
=================

Summary
-------

Performance in evaluating lattice constants for 23 solids, including pure elements,
binary compounds, and semiconductors.


Metrics
-------

1. MAE (Experimental)

Mean lattice constant error compared to experimental data

For each formula, a bulk crystal is built using the experimental lattice constants and
lattice type for the initial structure. This structure is optimised for each model
using the LBFGS optimiser, with the FrechetCellFilter applied to allow optimisation of
the cell, until the largest absolute Cartesian component of any interatomic force is
less than 0.03 eV/Å. The lattice constants of this optimised structure are then
compared to experimental values.


2. MAE (PBE)

Mean lattice constant error compared to PBE data

Same as (1), but optimised lattice constants are compared to reference PBE data.


Computational cost
------------------

Low: tests are likely to less than a minute to run on CPU.

Data availability
-----------------

Input structures:

* Built from experimental lattice constants from various sources

Reference data:

* Experimental data same as input data

* DFT data

  * Batatia, I., Benner, P., Chiang, Y., Elena, A.M., Kovács, D.P., Riebesell, J.,
    Advincula, X.R., Asta, M., Avaylon, M., Baldwin, W.J. and Berger, F., 2025. A
    foundation model for atomistic materials chemistry. The Journal of Chemical
    Physics, 163(18).
  * PBE-D3(BJ)
