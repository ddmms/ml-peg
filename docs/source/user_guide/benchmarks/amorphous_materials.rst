===================
Amorphous materials
===================

Amorphous carbon melt-quench
============================

Summary
-------

Assess the ability of models to reproduce amorphous carbon structure by running
melt-quench simulations at fixed density and measuring the sp3 fraction. Results are
compared against DFT and experimental reference curves from the literature. [#dft]_ [#expt]_
The DFT reference uses spin-paired DFT within the local density approximation (LDA). [#dft]_

Metrics
-------

1. MAE vs DFT

Mean absolute error in sp3 fraction relative to the DFT reference curve.

2. MAE vs Expt

Mean absolute error in sp3 fraction relative to the experimental reference curve.

Computational cost
------------------

High: melt-quench MD runs for each model and density grid point.

Data availability
-----------------

Reference DFT and experimental datasets are stored with the benchmark analysis files
and digitized from published curves. [#dft]_ [#expt]_

References
----------

.. [#dft] Jana *et al.* (2019), Modelling Simul. Mater. Sci. Eng. 27 085009.
   DOI: 10.1088/1361-651X/ab45da

.. [#expt] Deringer & Csányi (2017), Phys. Rev. B 95, 094203.
   DOI: 10.1103/PhysRevB.95.094203
