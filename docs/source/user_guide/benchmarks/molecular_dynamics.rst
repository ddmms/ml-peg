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

Benchmark speed
------------------

Multi-day: tests take several days to run on GPU.

Data availability
-----------------

Input structures:

* Weber et al., Efficient Long-Range Machine Learning Force Fields for
    Liquid and Materials Properties.
    arXiv:2505.06462 [physics.chem-ph]

Reference data:

* Same as input data
* Experimental


Polymer densities
=================

Summary
-------

Performance in predicting room-temperature amorphous densities for 130
polymers. Starting structures are prebuilt polymer cells, and the reference
data are experimental densities.

Metrics
-------

1. Density error

For each polymer, the density is calculated by averaging over the final 500 ps
NPT production stage of a 24-stage Polymatic-style equilibration protocol. The
protocol uses a 0.5 fs timestep and runs for about 2.06 ns per model and
polymer at the default time prefactor. The predicted density is compared to the
experimental reference density.

Computational cost
------------------

Very high: tests are likely to take several days to run on GPU.

Data availability
-----------------

Input structures:

* Built from the SimPoly polymer dataset and stored as prebuilt structures for
  ML-PEG calculations.

Reference data:

* Simm et al., SimPoly: Simulation of Polymers with Machine Learning Force
  Fields Derived from First Principles. arXiv:2510.13696 [physics.comp-ph]
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

Benchmark speed
------------------

Multi-day: tests take several days to run on GPU.

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

Benchmark speed
------------------

Multi-day: tests take several days to run on GPU.


Data availability
-----------------
Input structures:
Packmol generated

Reference data:
* M. Southard and D. Green, Perry’s Chemical Engineers’ Handbook, 9th Edition. McGraw-Hill Education, 2018.
* Experimental


Bond length distribution
========================

Summary
-------

Performance in maintaining physically reasonable covalent bond lengths during molecular
dynamics of small organic molecules. For each of a set of molecules covering the C-C, C=C,
C#C, C-N, C-O, C=O and C-F bond types, an NVT molecular dynamics simulation is run at 300 K
starting from a QM-optimised reference geometry, and the deviation of a tracked bond from
its reference length is measured along the trajectory.

Metrics
-------

1. Bond length deviation

The length of the tracked bond is measured at each frame of the trajectory, and its absolute
deviation from the reference bond length is averaged over the trajectory and across all
molecules. A well behaved potential keeps bonds close to their reference length, so a lower
deviation is better.

A histogram shows the distribution of the sampled bond length deviations for each model.

Computational cost
------------------

High: 8 molecules of 4-13 atoms, one MD simulation each, of 1,000,000 steps, i.e. 1 ns at a
1 fs timestep. The molecules are small, so the cost per step is dominated by per-call
overhead rather than by system size, and tests are likely to take a couple of hours per
model on GPU. Faster inference can be achieved using the jax-accelerated simulations in
MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. Reference geometries selected from the QM9 dataset
  (Ramakrishnan et al., Scientific Data 1, 140022, 2014).

Reference data:

* QM-optimised equilibrium bond lengths of the reference geometries.


Ring planarity
==============

Summary
-------

Performance in maintaining planar aromatic rings during molecular dynamics simulations
of small organic molecules, evaluated on 6 molecules with aromatic ring systems
selected from the QM9 dataset. For each molecule, an NVT molecular dynamics simulation
is run at 300 K starting from the QM9-optimised geometry, and the deviation of
the ring atoms from their best-fit plane is measured along the trajectory.

Metrics
-------

1. Planarity deviation

At each frame of the trajectory, the ring atoms are fitted to a plane and the root mean
square deviation of the atoms from that plane is calculated. This is averaged over the
trajectory and across all molecules. Aromatic rings are planar, so a well-behaved
potential keeps this deviation small; a lower deviation is better. If any molecule is
unstable (fails with an error or the simulation explodes), the metric is reported as
NaN, as is the benchmark score.

2. Planarity deviation (stable only)

The planarity deviation, averaged only over the molecules whose simulations were
stable. This is identical to the metric above unless any molecule was unstable, in which
case it still gives a value while the metric above is NaN. It is reported for
information only and does not contribute to the benchmark score.

A histogram shows the distribution of the sampled planarity deviations for each model,
using only stable simulations.

Computational cost
------------------

High: 6 molecules of 12-16 atoms, one MD simulation each, of 1,000,000 steps, i.e. 1 ns at a
1 fs timestep. The molecules are small, so the cost per step is dominated by per-call
overhead rather than by system size, and tests are likely to take a couple of hours per
model on GPU. Faster inference can be achieved using the jax-accelerated simulations in
MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. Starting geometries selected from the QM9 dataset
  (Ramakrishnan et al., Scientific Data 1, 140022, 2014).

Reference data:

* QM-optimised starting geometries from QM9 (:math:`B3LYP/6-31G(2df,p)` level of theory).


Stability
=========

Summary
-------

Performance in running stable molecular dynamics across a diverse set of
systems: small molecules (containing H/C/N/O, sulfur, and halogens), peptides
in vacuum (neurotensin, PDB: 2LNF; cyclic oxytocin, PDB: 7OFG), a protein in
vacuum (PDB: 1A7M), and solvated peptides with and without counter-ions.
Each system is run with a 100,000 step NVT molecular dynamics simulation at 300 K,
with a frame saved every 100 steps.

Metrics
-------

1. Stability score

The stability score as computed by MLIP Audit. Computed as:

.. math::

   S =
   \begin{cases}
   0, & \text{the simulation failed with an error} \\
   \frac{1}{2} \frac{f_\mathrm{e}}{N}, & \text{the simulation exploded} \\
   \frac{1}{2} + \frac{1}{2} \frac{f_\mathrm{h}}{N}, &
   \text{no explosion, but a hydrogen was lost} \\
   1, & \text{stable for the whole trajectory}
   \end{cases}

where :math:`N` is the total number of frames the simulation is run for,
:math:`f_\mathrm{e}` is the frame where the system exploded, and
:math:`f_\mathrm{h}` is the frame where a hydrogen drifted at least 2.5 Å away
from all heavy atoms. The reported value is the mean of :math:`S` over all systems.
The mean is between 0 and 1, where higher is better.

A scatter plot below the table shows the stability score for each system and
model. Individual models can be toggled via the legend.

Computational cost
------------------

High: tests are likely to take many hours on GPU. Faster simulation times can be
achieved using the jax accelerated simulations in MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep.
  Structures derived from PDB entries 2LNF, 7OFG, and 1A7M.
