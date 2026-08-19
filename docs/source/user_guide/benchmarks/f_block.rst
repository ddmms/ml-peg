=======
f-block
=======

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
relative energies reported in the reference dataset.


Computational cost
------------------

Low: tests are likely to take less than a minute to run on CPU.


Data availability
-----------------

Input structures:

* T. Rose, M. Bursch, J.-M. Mewes, and S. Grimme, Fast and Robust Modeling of
  Lanthanide and Actinide Complexes, Biomolecules, and Molecular Crystals with
  the Extended GFN-FF Model, Inorganic Chemistry 63 (2024) 19364-19374.

Reference data:

* Relative isomer energies from r2SCAN-3c (see Supporting Information of the
  above reference).


Plutonium Dioxide
=================

Summary
-------

General performance on Plutonium Dioxide against DFT+U calculations. The DFT+U calculations are evaluted on samples in the temperature range 0-1200K and been have parameterized to correctly predict the lattice constant (within 0.3%) and thermal expansion at low temperature.

Metrics
-------

1. Energy MAE (PBE+U)

Mean absolute error of energy predictions (per atom).

2. Force MAE (PBE+U)

Mean absolute error of force (individual components) predictions against DFT+U calculations.

3. Stress MAE (PBE+U)

Mean absolute error of stress (individual tensor components) predictions against DFT+U calculations.

Computational cost
------------------

Low

Data availability
-----------------

Reference data: availabile in repo. Data and complete calculation details will be released in an upcoming publication. For now, please contact willdavie2002@gmail.com.
