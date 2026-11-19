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
---------------

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

Benchmark speed
---------------

Multi-day: tests are likely to take several days to run on GPU.

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
---------------

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
---------------

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

Benchmark speed
---------------

Slow: 8 molecules of 4-13 atoms, one MD simulation each, of 1,000,000 steps, i.e. 1 ns at a
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

Benchmark speed
---------------

Slow: 6 molecules of 12-16 atoms, one MD simulation each, of 1,000,000 steps, i.e. 1 ns at a
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

Benchmark speed
---------------

Very slow: tests are likely to take many hours on GPU. Faster simulation times can be
achieved using the jax accelerated simulations in MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep.
  Structures derived from PDB entries 2LNF, 7OFG, and 1A7M.


Inference speed
===============

Summary
-------

Measure of model inference speed and how it scales with system size. A
size-stratified dataset of 13 biomolecular structures (71-6714 atoms) is used.
For each system the model forward pass (energy + forces) through the model's
ASE calculator is timed.

Metrics
-------

1. Forward time / atom

For each structure, the mean wall-clock time of a single model forward pass
(energy + forces) is measured, excluding warm-up passes, and divided by the number of
atoms. The reported metric is the mean of this per-atom forward time across
structures, omitting the slowest 20% of passes, in microseconds per atom. Lower is faster.
If the forward pass fails for any structure, the metric is reported as NaN.

Models are timed through their ASE calculators in low (float32) precision, as used for
the molecular dynamics benchmarks. Each timed pass forces a full ``calculate()`` call,
so the timing can include the calculator's own neighbour-list construction and per-call
overhead. By contrast, MLIP Audit times its native force fields on the pure network
forward pass over a pre-built graph. The two paths measure different work, so these
timings should not be compared directly with MLIP Audit's own inference-speed results.

The accompanying scaling plot shows the model throughput (number of atoms divided by the
forward-pass time, in atoms per second) against the number of atoms on log-log axes,
with a power-law fit per model.

This is a wall-clock, hardware-dependent measurement rather than a level of theory:
results are only comparable across models run on the same hardware. The reference
hardware is an NVIDIA H100 GPU.

Benchmark speed
---------------

Low: tests are likely to take minutes to run on GPU.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. A size-stratified set of 13 structures
  taken from the PDB (elements C, N, H, O, S, P): mostly protein chains, plus a
  peptide and an RNA fragment.

Reference data:

* Not applicable: this is a wall-clock hardware measurement, not a comparison to a
  reference level of theory.


Aluminosilicates Densities
==========================

Summary
-------

Benchmark evaluating the density prediction of three aluminosilicate glasses
(albite, anorthite, and sanidine) against experimental data. A melt-quench
procedure is applied starting from randomly generated structures at 2500 K and
quenching to 300 K in the NPT ensemble.

Metrics
-------

For each composition (albite, anorthite, sanidine) and for the mean over the three:

1. Density MAE (g/cm^3): mean of the absolute errors of the individual
   replicas with respect to the experimental density.
2. Density MAPE (%): mean of the absolute percentage errors of the individual
   replicas.

Errors are computed for each replica and then averaged, not on the mean density.

Thresholds: a model is considered to reproduce the density correctly when the
MAE is 0.05 g/cm^3 (MAPE 2 %) or lower which is of the order of the spread
between replicas of the same model. A model is considered completely inaccurate
when the MAE is 0.30 g/cm^3 (MAPE 15 %) or higher, which is well above
the 5-10 % systematic underestimation typical of MD melt-quench simulations.
This underestimation is expected, because the quench rate (5 K/ps) is many orders
of magnitude faster than in experiment.

Methodology
-----------

For each system, a randomly generated structure first undergoes a 5 ps simulation
in the NVT ensemble at 2500 K, followed by 20 ps at 2500 K in the NPT ensemble,
using Berendsen thermostat and barostat. The structure is then quenched from
2500 K down to 300 K at 5 K/ps. Finally, it is equilibrated at 300 K for 20 ps.
The reported density is the average over 21 snapshots, one every 1000 steps (1 ps),
saved during the final equilibration stage. Three replicas (different starting
structures) are run for each composition.

A similar melt-quench protocol was established on larger simulation boxes in the
literature (https://doi.org/10.1111/jace.70962). The choice of a ~350-atom system
represents a pragmatic compromise to significantly reduce computational cost
while retaining physically meaningful glass density predictions.

The MD uses float32 for the MACE models and MatterSim. The other models use the
default ML-PEG precision. `mace-omol` and `mace-polar-1-l` did not complete the
protocol: they performed only a few tens of MD steps in several hours, while the
full protocol requires 485,000 steps. They are therefore reported without results.

Benchmark speed
---------------

Very low. For a single run (485 ps, ~350 atoms) with `mace-matpes-r2scan` on
one NVIDIA A100 64 GB GPU (Leonardo booster, CINECA), the runtime is about
29 h. Per model, 9 runs are needed (3 compositions x 3 replicas).

Data availability
-----------------

Input structures:

Generated with Packmol using `data/build_structure.py`, which is included in the
benchmark folder. Three random structures (seeds 0, 1 and 2) are generated for each
composition.

Reference data:

* Experimental glass densities of albite, anorthite and sanidine from
  M. Taylor and G. E. Brown, Structure of mineral glasses - I. The feldspar glasses
  NaAlSi3O8, KAlSi3O8, CaAl2Si2O8.
  Geochim. Cosmochim. Acta 43, 61-75 (1979),
  https://doi.org/10.1016/0016-7037(79)90047-4
