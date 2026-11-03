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


Binary mixtures
===============

Summary
-------

Performance in predicting the enthalpy of mixing (ΔHmix), mixture density, excess molar
volume (V^E) and pure-liquid densities of 10 binary liquid mixtures, at 3 compositions
each (30 mixture boxes and 20 pure-liquid boxes). The systems are drawn from the OpenFF
Sage force-field training and validation data and cover hydrogen-bonded, polar, aromatic
and non-polar interactions, including two aqueous mixtures. Five mix exothermically
(ΔHmix < 0) and five endothermically (ΔHmix > 0):

.. list-table::
   :header-rows: 1

   * - Mixture
     - Interactions
     - ΔHmix
     - T (K)
     - Scored properties
   * - pyrrole + pyridine
     - N–H···N hydrogen bonds between the components
     - < 0
     - 298.15
     - ΔHmix, density, V^E
   * - methanol + water
     - Hydrogen-bonded, aqueous
     - < 0
     - 298.15
     - ΔHmix, density, V^E
   * - water + pyridine
     - Hydrogen-bonded, aqueous
     - < 0
     - 303.15
     - ΔHmix only (no experimental density)
   * - NMP + 1,2-dichloroethane
     - Polar aprotic with a weak C–H donor
     - < 0
     - 298.15
     - ΔHmix, density, V^E
   * - acetone + N-methylaniline
     - N–H···O=C hydrogen bonds between the components
     - < 0
     - 298.15
     - ΔHmix, density, V^E
   * - 1-propanol + benzene
     - Alcohol hydrogen bonds lost on dilution; benzene only a weak (O–H···π) acceptor
     - > 0
     - 298.15
     - ΔHmix, density, V^E
   * - tetrahydropyran + ethanol
     - Alcohol hydrogen bonds partly replaced by weaker O–H···O(ether) bonds
     - > 0
     - 298.15
     - ΔHmix, density, V^E
   * - dipropylamine + DMF
     - Weak amine self-association with a polar amide
     - > 0
     - 298.15
     - ΔHmix, density, V^E
   * - propionitrile + tert-amyl ethyl ether
     - Dipolar nitrile with an ether
     - > 0
     - 298.15
     - ΔHmix, density, V^E
   * - heptane + toluene
     - Non-polar
     - > 0
     - 298.15
     - ΔHmix, density, V^E

Boxes contain 100 to 521 molecules (952 to 2,400
atoms), with edges of at least 25 Å so that periodic images stay outside the receptive
field of typical MLIPs. Each box is built from low-energy conformers, annealed and
pre-equilibrated with OpenFF Sage (openff-2.3.0) at the experimental density.

Each box is simulated once with NPT MD (isotropic MTK barostat, 1 fs timestep, 1 atm)
for 0.2 ns of equilibration followed by 1 ns of production.

Metrics
-------

Every property is scored by its mean absolute z-score (MAZE). The mean absolute error
(MAE) is shown alongside it, at weight 0.

1. ΔHmix MAE (kJ/mol) and MAZE, over the 30 mixture compositions

   ΔHmix = h_mix - x h_A - (1 - x) h_B, where h = (⟨E_pot⟩ + p⟨V⟩) / N_molecules.
   The experimental value is interpolated to each box's exact composition with a
   Redlich-Kister fit to the full isotherm.

2. Mixture density MAE (g/cm³) and MAZE, over the 27 compositions with experimental
   mixture densities.

3. V^E MAE (cm³/mol) and MAZE, over the same 27 compositions.

4. Pure density MAE (g/cm³) and MAZE, over the 7 pure liquids not already scored by the
   liquid densities benchmark.

   For every MAZE, each error is divided by a fixed reference uncertainty
   σ_ref = sqrt(σ_exp² + σ_Sage²), where σ_Sage is the standard deviation of 5
   independent Sage runs of the same boxes with the same protocol (pooled per system and
   floored at the median over systems). σ_ref does not depend on the model, so models
   are not rewarded for noisy predictions.

   'Good' thresholds are the score expected for a model that matches experiment exactly,
   given the noise of one 0.2 + 1 ns run per box, plus two standard deviations. 'Bad'
   thresholds are five times the score of Sage, which was trained on these properties.

5. Flagged boxes (not scored)

   The number of boxes with fewer than 100 effective samples, a significant drift or
   difference between the halves of production (block averages, Holm-corrected,
   family-wise p < 0.05), a change in covalent bonding, or a like-neighbour fraction
   well above that of the Sage runs (demixing).

Uncertainties
-------------

Mean values and their standard errors are calculated with the autocorrelation-based
statistical inefficiency (Chodera et al.) on every logged sample of the production data.
Confidence intervals for the metrics are bootstrapped over the 50 boxes, so the
correlation between compositions that share a pure-liquid box is included. Each model
is run once per box. In the Sage reference runs, the spread between independent
replicates was about 1.17 times the within-run standard error for the enthalpy (1.06
times for the density), so single-run confidence intervals are likely to be somewhat too
narrow.

Sage reference
--------------

OpenFF Sage (openff-2.3.0) results, averaged over 5 replicates, are shown as a reference
trace in the composition plots. They give a ΔHmix MAE of 0.59 kJ/mol (MAZE 5.0), a
mixture density MAE of 0.0094 g/cm³ (MAZE 10.1) and a V^E MAE of 0.24 cm³/mol
(MAZE 1.9). Sage misses the strongly exothermic, hydrogen-bonded mixtures pyrrole +
pyridine and water + pyridine (the latter with the wrong sign).

Limitations
-----------

* Nuclei are classical, so quantum nuclear effects on hydrogen bonding (relevant for the
  aqueous systems) are not included.
* There is no N-H amide, amide-water or ester system.
* The systems come from the Sage training and validation data, rather than a directly
  designed spread of chemistry.
* The noise behind σ_ref and the thresholds comes from Sage, which may have different
  correlation times from MLIPs.

Benchmark speed
------------------

Multi-day: 50 boxes × 1.2 ns (76,065 atoms in total), about 140 MACE-OFF23(M) A100
GPU-days per model, the most expensive benchmark in this category.

Data availability
-----------------

Input structures:

* Built with packmol and pre-equilibrated with OpenFF Sage (openff-2.3.0, NAGL AM1-BCC
  charges, TIP3P water). The selection, build, Sage reference and power-analysis scripts
  are kept in the separate
  `ml-peg-liquid-mixtures-input-generation <https://github.com/fjclark/ml-peg-liquid-mixtures-input-generation>`_ repository.

Reference data:

* Experimental, from the NIST TRC ThermoML archive, via the OpenFF Sage 2.0 and Sage 2.3
  physical-property data sets (openforcefield/openff-sage, openforcefield/ash-sage-rc2).
  Full isotherms and pure densities are taken from each source paper and water densities
  are from IAPWS-95.
* The full reference list, with the properties and systems taken from each source, is
  in `sources.csv <https://github.com/fjclark/ml-peg-liquid-mixtures-input-generation/blob/main/selection/sources.csv>`_.


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
