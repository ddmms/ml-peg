===========
Lanthanides
===========

Isomer complexes
================

Summary
-------

Performance in predicting relative isomer energies for lanthanide complexes
compared to r2SCAN-3c DFT reference data.


Metrics
-------

1. Relative isomer energy MAE

Accuracy of relative isomer energy predictions.

For each complex, the relative isomer energies are computed with respect to the
lowest-energy isomer in the r2SCAN-3c reference set and compared to the r2SCAN-3c
relative energies reported in the reference dataset. The r2SCAN-3c geometries are
used, with wB97X-V/def2-mTZVPP single-point calculations reported for validation
in the source study.


Computational cost
------------------

Low: tests are likely to take less than a minute to run on CPU once model outputs
are available.


Data availability
-----------------

Input structures:

* T. Rose, M. Bursch, J.-M. Mewes, and S. Grimme, Fast and Robust Modeling of
  Lanthanide and Actinide Complexes, Biomolecules, and Molecular Crystals with
  the Extended GFN-FF Model, Inorganic Chemistry 63 (2024) 19364-19374.

Reference data:

* Relative isomer energies from r2SCAN-3c (see Supporting Information of the
  above reference).
