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


Aluminosilicates Densities
============================


Summary
-------

Benchmark evaluating the density prediction of three aluminosilicate glasses (albite, anorthite, and sanidine) against experimental data.
A melt-quench procedure is applied starting from randomly generated structures at 2500 K and quenching to 300 K in the NPT ensemble.

Metrics
-------

1. MAE of density for individual systems and averaged overall.
2. MAPE of density for individual systems and averaged overall.


Methodology
-----------

For each system, a randomly generated structure first undergoes a 5 ps simulation in the NVT ensemble at 2500 K, followed by 20 ps at
2500 K in the NPT ensemble. The structure is then quenched at a cooling rate of 5 K/ps from 2500 K down to 300 K. Finally, it is
equilibrated at 300 K for 20 ps. The reported density is averaged across 20 snapshots saved every 1 ps during the final equilibration
stage and compared against experimental reference values.
A similar melt-quench protocol was established on larger simulation boxes in literature (doi.org/10.1111/jace.70962). The choice of a ~350-atom system represents a pragmatic compromise to significantly reduce computational cost while retaining physically meaningful glass density predictions.


Computational cost
------------------

Very high (simulations require several days to complete on a GPU).


Data availability
-----------------
Input structures:
Generated using Packmol.

Reference data:
* Taylor, M. & Brown, G.E. (1979), Structure of mineral glasses -- I.
* The feldspar glasses NaAlSi3O8, KAlSi3O8, CaAl2Si2O8.
* Geochim. Cosmochim. Acta, doi:10.1016/0016-7037(79)90047-4
