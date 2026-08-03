==================
Molecular dynamics
==================

Liquid densities
================

Summary
-------

Performance in predicting densities for 61 organic liquids, each system consisting of
about 1000 atoms. The dataset covers aliphatic, aromatic molecules, as well as different
functional groups and halogenated molecules.

Metrics
-------

1. Density error

For each system, the density is calculated by taking the average density of an NPT molecular
dynamics run. The initial part of the simulation, here 500 ps, is omitted from the density
calculation. This is compared to the reference density, obtained from experiment.

Computational cost
------------------

Very high: tests are likely to take several days to run on GPU.

Data availability
-----------------

Input structures:

* Weber et al., Efficient Long-Range Machine Learning Force Fields for
    Liquid and Materials Properties.
    arXiv:2505.06462 [physics.chem-ph]

Reference data:

* Same as input data
* Experimental


Water density
=============

Summary
-------

Performance in predicting the density of water at temperatures of 270, 290, 300, and 330 K.
The water systems consist of 333 molecules.

Metrics
-------

1. Density error

For each system, the density is calculated by taking the average density of an NPT molecular
dynamics run. The initial part of the simulation, here 500 ps, is omitted from the density
calculation. This is compared to the reference density, obtained from experiment.

Computational cost
------------------

Very high: tests are likely to take several days to run on GPU.

Data availability
-----------------

Input structures:

* Weber et al., Efficient Long-Range Machine Learning Force Fields for
  Liquid and Materials Properties. arXiv:2505.06462 [physics.chem-ph]

Reference data:

* Same as input data
* Experimental


Water ethanol density curves
============================

Summary
-------

Benchmark of the density of water-ethanol mixtures for different concentrations of ethanol, compare to experiment.
1 ns of NPT MD on about 120 water/ethanol molecules for 6 concentrations.


Metrics
-------

1. rms of the density difference.
2. rms of the excess volume difference.
3. Concentration of the minimal excess volume.


For each system, the density is calculated by taking the average density of an NPT molecular
dynamics run. The initial part of the simulation, here 500 ps, is omitted from the density
calculation. This is compared to the reference density, obtained from experiment.
The excess volume is computed as the difference between the actual molar volume of the mixture and the ideal molar volume obtained by linear combination of the pure-component molar volumes.
The concentration of the minimal excess volume is estimated by fitting a quadratic to the three grid points surrounding the minimum and taking the vertex of the parabola.

Computational cost
------------------

Very high: tests are likely to take several days to run on GPU.


Data availability
-----------------
Input structures:
Packmol generated

Reference data:
* M. Southard and D. Green, Perry’s Chemical Engineers’ Handbook, 9th Edition. McGraw-Hill Education, 2018.
* Experimental

Battery electrolyte densities
=============================

Summary
-------

Performance in predicting the densities of 25 battery electrolyte systems at
298.2 K: six neat glyme and carbonate solvents (DME, DEGDME, TEGDME, PC, DEG,
DMC), and 19 electrolytes formed from NaPF6, NaOTf, NaTFSI and KPF6 at 0.1, 0.5
and 1.0 M. Systems contain roughly 500-2200 atoms.

Because the same salts appear across several solvents and concentrations, the
benchmark separates general liquid-density performance from the harder question
of whether a potential handles concentrated ionic environments.

Metrics
-------

1. Density MAE
2. Density RMSE
3. Density MAPE

For each system, the density is the average over an NPT molecular dynamics run
at 298.2 K and 1 atm with a 1 fs timestep. The first 50 ps of each 150 ps
trajectory is discarded, leaving 100 ps of production. Inputs are already
NPT-equilibrated, which is why a shorter equilibration period is used than for
the organic liquid densities benchmark. Densities are compared to experimental
values at 298.2 K.

Computational cost
------------------

High: 150 ps of NPT MD per system, for 25 systems per model.

Data availability
-----------------

Input structures:

* Kumar et al., Prediction and Experimental Verification of Electrolyte
  Solvation Structure from an OMol25-Trained Interatomic Potential.
  arXiv:2603.20183 [physics.chem-ph]
* https://github.com/KMNitesh05/sodium-ion-battery-electrolyte-dataset

Reference data:

* Same as input data
* Experimental
