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


NVE energy conservation
=======================

Summary
-------

Performance in conserving the total mechanical energy ``E = PE + KE`` during
microcanonical (NVE) molecular dynamics. Four systems are simulated: a 46 atom organic
molecule in vacuum, a periodic box of 500 water molecules, and two solvated peptides
(Oxytocin, and Neurotensin with counter-ions), spanning 46 to 2642 atoms. Each runs for
50,000 velocity-Verlet steps at a 1 fs timestep, i.e. 50 ps, with velocities initialised
at 300 K and no thermostat.

Without a thermostat, the total energy is a conserved quantity of the dynamics, so this
benchmark needs no external reference: any systematic drift is a defect of the potential
energy surface or of the consistency between the energies and forces derived from it.

Metrics
-------

1. Energy drift
2. Energy drift ratio
3. Systems completed
4. NVE score

The total energy is recorded at every snapshot and its deviation from the first frame is
fitted with a straight line. The fitted slope, divided by the number of atoms, gives the
energy drift in meV/atom/ps, which is the form usually quoted in the literature and is
comparable between the four systems. Dividing the fitted drift over the whole trajectory
by the standard deviation of the kinetic energy instead gives the dimensionless energy
drift ratio, where a value of 1 means the accumulated drift has grown as large as the
natural kinetic energy fluctuation of the system. The NVE score maps that ratio through
MLIP Audit's soft threshold at 1 and averages over the systems, and is the only scored
metric; the two drift columns and the count of completed systems are reported for
information.

Both drift metrics average over the systems that completed, so the number of systems
completed is reported alongside them: a model that only manages one of the four is not
comparable to one that manages all four. Note also that a trajectory which is stopped
early, because the simulation exploded, is fitted over the shorter window it produced.

Clicking a row shows the total energy drift per atom along the trajectory for each of
that model's systems, keeping its sign, so heating can be told apart from cooling.

Computational cost
------------------

High: 4 systems of 46 to 2642 atoms, one MD simulation each, of 50,000 steps, i.e. 50 ps
at a 1 fs timestep. That is 200,000 force evaluations per model, and three of the four
systems are periodic and over 1300 atoms, so tests are likely to take many hours to run
on GPU. The simulations are driven through ASE, one calculator call per step, with no
jit and no episode batching; faster inference can be achieved using the jax-accelerated
simulations in MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. The equilibrated 500-molecule water box is
  shared with the water radial distribution benchmark. The peptides are Oxytocin
  (PDB: 7OFG) and Neurotensin with counter-ions (PDB: 2LNF).

Reference data:

* None. Energy conservation is an internal consistency property of the potential, so
  there is no reference method to compare against.
