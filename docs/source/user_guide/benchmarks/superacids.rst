==========
Superacids
==========

HF/SbF5 mixture densities
=========================

Summary
-------

Performance in predicting the liquid density of HF/SbF5 mixtures, the archetypal
superacid system, at three compositions: pure HF, a 10 mol % SbF5 mixture, and pure
SbF5. Each system consists of about 200 atoms, simulated at 288.65 K and 1 atm.

Metrics
-------

1. MAPE

For each composition, the density is calculated from the average volume of an NPT
molecular dynamics run of 100 ps. The first half of the simulation is discarded as
equilibration. The mean absolute percentage error over the three compositions is
compared to the reference densities, obtained from experiment.

Computational cost
------------------

High: about 8 GPU hours per model for all three compositions, for a model that
includes a dispersion correction.

Data availability
-----------------

Input structures:

* Generated with Packmol, available from the ML-PEG S3 bucket

Reference data:

* Shair and Schurig, Vapor-Liquid Equilibrium of Antimony Pentafluoride-Hydrogen
  Fluoride. Ind. Eng. Chem. 43, 1624 (1951). https://doi.org/10.1021/ie50499a042
* Experimental


HF structure factor
===================

Summary
-------

Performance in predicting the total neutron structure factor of liquid HF, which probes
the hydrogen-bonded chain structure of the liquid. A system of 100 HF molecules is
simulated at 296 K and 1.2 bar for 50 ps of NPT molecular dynamics.

Since the experimental reference is measured on the deuterated liquid, all H are
transmuted to D before the structure factor is computed. S(q) is obtained with MDANSE,
as the Fourier transform of the pair distribution function weighted by coherent
scattering lengths, using the second half of the trajectory sampled every other frame.
The real-space cutoff is kept the same for every model, so that the transform is
truncated identically.

Metrics
-------

1. S(q) R-factor

The relative deviation of the calculated structure factor from experiment,
``sum|S_exp - S_calc| / sum|S_exp|``, evaluated from the first experimental point to
4 1/A. Calculated and experimental structure factors are computed on the same grid of
scattering vectors.

2. First peak position error

The absolute error in the position of the maximum of S(q), located over the same range
as the R-factor. The position is read directly off the grid of scattering vectors,
whose spacing is 0.05 1/A.

The "good" threshold of the R-factor is the statistical noise of the protocol itself,
measured by splitting the production window of one model into two halves.

Computational cost
------------------

High: about 6 GPU hours per model, for a model that includes a dispersion correction.

Data availability
-----------------

Input structures:

* Generated with Packmol, available from the ML-PEG S3 bucket

Reference data:

* McLain, Benmore, Siewenie, Urquidi and Turner, On the Structure of Liquid Hydrogen
  Fluoride. Angew. Chem. Int. Ed. 43, 1952 (2004).
  https://doi.org/10.1002/anie.200353289
* Experimental
