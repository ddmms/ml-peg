===========
Physicality
===========

Locality
========

Summary
-------

Performance in respecting locality, by measuring deviations from a neglible interactions
between acetone and distance atoms.

Metrics
-------

1. Maximum difference in force due to "ghost atoms"

Forces on an isolated acetone molecule are calculated, and the forces on the same atoms
are calculated after 20 Ne atoms are placed in a 60 Å cubic box, at least 40 Å from the
acetone's centre of mass. The magnitude of the maximum difference in force is reported.

2. Mean difference in force due to a distance hydrogen

Forces on an isolated acetone molecule are calculated, and the forces on the same atoms
are calculated after a single hydrogen atom is placed between 20 and 50 Å from the
acetone's centre of mass. This is repeated for 30 different random placements of the
hydrogen atom, the mean force difference on the acetone atoms is calculared.

3. Standard deviation in force due to a distance hydrogen

Same as (2), but the standard deviation of the force difference on the acetone atoms is
calculated.


Computational cost
------------------

Low: tests are likely to take less than a minutes to run on CPU.


Data availability
-----------------

None required.


Extensivity
===========

Summary
-------

Performance in respecting extensivity, by measuring differences in energy between
isolated systems, and the same systems combined, but significantly separated.

Metrics
-------

1. Absolute energy difference between isolated and combined slabs

The energy of two isolated slabs is calculated, and the energy of the combined system,
with the two slabs separated by 100 Å is calculated. The absolute energy difference
between the sum of the isolated slabs and that of the combined system is calculated.


Computational cost
------------------

Low: tests are likely to take less than a minutes to run on CPU.


Data availability
-----------------

None required.


Diatomics
=========

Summary
-------

This benchmark probes the short- to medium-range behaviour of every homonuclear and
heteronuclear diatomic pair in the periodic table. Each MLIP is evaluated on a 100-point
linear distance grid spanning 0.18-6.0 Å and the resulting energies and projected forces
are analysed for unphysical oscillations.

Metrics
-------

1. Force flips

   Average number of times the projected bond force changes sign. Forces are projected
   onto the bond axis and values below :math:`10^{-2}` eV/Å are rounded to zero to avoid
   counting noise-induced flips. A smooth curve should switch from attraction to repulsion
   only once at the minimum.


2. Energy minima

   Mean count of distinct minima in the energy-distance profile. Local minima are
   found from the second derivative, where a physical diatomic should show a single
   minimum.


3. Energy inflections

   Mean number of inflection points obtained from the second derivative of the energy
   curve. Inflections are flagged when the second derivative changes sign with a
   tolerance of 0.5 eV/Å² to avoid counting noise-induced inflections. A physical diatomic
   curve should show one inflection point.

4. :math:`\rho(E, \text{repulsion})`

   Spearman correlation between atomic separation and energy on the repulsive side of the well
   (bond lengths ≥ the equilibrium spacing). A perfect diatomic curve should show a strong
   negative correlation, so a value of -1, indicating that as atoms get further apart, the energy
   decreases.

5. :math:`\rho(E, \text{attraction})`

   Spearman correlation between distance and energy on the attractive side (bond lengths
   shorter than the equilibrium spacing). A perfect diatomic curve should show a strong
   positive correlation, so a value of +1, indicating that as atoms get closer together, the
   energy increases.

Computational cost
------------------

High: Expected to take hours to run on GPU, or around one day for slower MLIPs.

Data availability
-----------------

None required; diatomics are generated in ASE.


Oxidation States
================

Summary
-------

Examines the model's ability to capture different oxidation states of Fe in aqueous solution [1, 2]. Two systems containing Fe 2Cl (Fe+2 state) and Fe 3Cl (Fe+3 state) in water are simulated at 300K for 20ps with NVT MD.
The solvation cell is expected to be tighter for iron in the Fe+3 state. This effect, if an MLIP can correctly capture the Fe oxidation state, appears as a split on the Fe-O RDF peaks.
This test examines whether a split appears between the Fe-O RDF peaks of the two systems. Additionally, the benchmark examines whether the peaks fall into the expected experimental range [1].

[1] Kocer, Emir, et al. "Machine learning potentials for redox chemistry in solution." arXiv preprint arXiv:2410.03299 (2024).
[2] Batatia, Ilyes, et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry." arXiv preprint arXiv:2602.19411 (2026).

Metrics
-------

1. Fe-O RDF Peak Split

The similarity of the aqueous Fe 2Cl and Fe 3Cl system RDFs is examined.
If a split is present the score is +1 and in the case there is no clear split the score is 0.
This metric determines whether a model can capture the different oxidation states of Fe and is therefore weighted as 5x more important than the two following metrics.


2. Fe +2 Peak Experimental Ref Deviation

Deviation of the Fe 2Cl system's RDF peak position from the experimental range.


3. Fe +3 Peak Experimental Ref Deviation

Deviation of the Fe 3Cl system's RDF peak position from the experimental range.

Computational cost
------------------

High: Expected to take hours to run on GPU, or around one day for slower MLIPs.

Data availability
-----------------

Starting configurations for the MD are available on S3 bucket. Experimental reference ranges for the RDF peaks were taken from [1].

[1] Kocer, Emir, et al. "Machine learning potentials for redox chemistry in solution." arXiv preprint arXiv:2410.03299 (2024).


Water Slab Dipoles
==================

Summary
-------

Distribution of dipole of water slab, checking for width of distribution and structures with dielectric breakdown.


Metrics
-------

1. Standard Deviation of Dipole Distribution

For a number of samples from an MD simulation, the total dipole is calculated. Compare to a reference of a LR model trained on revPBE-D3.

2. Number of structures with dielectric breakdown

Estimate band gap based on dipole, count structures where band gap disappears.


Computational Cost
------------------

High: Requires around 500 ps of MD of 40 A slab to get converged distribution, around 1 day on one GPU.


Data availability
-----------------

https://arxiv.org/html/2603.04228v1


Water-Cl2 cluster relaxation
============================

Summary
-------

This test is mainly for long-range models to probe the stability of a Cl2 molecule when two solvated Cl- ions are present far outside the receptive field of the Cl2 molecule, and the total charge of the system being -2.

Geometry relaxation may lead to the Cl2 molecule dissociating (incorrect behaviour) or staying stable (correct behaviour).

Metrics
-------

1. Dissociation of the Cl-Cl bond based on the interatomic distance.

Computational Cost
------------------

Low: Requires up to 1000 optimizer steps for a 400-atom system, taking several GPU-minutes to complete.


Data availability
-----------------

The initial structures were generated for MACE-POLAR-1 https://arxiv.org/abs/2602.19411


Rotational Symmetry
===================

Summary
-------

Performance in respecting rotational symmetry. Under a rigid rotation of a structure the
energy is invariant and the forces are equivariant (they rotate with the structure), so
any deviation reflects the model's implementation rather than physics. Architectures
that predict forces directly, rather than as the gradient of the energy, are not
equivariant by construction, and this test measures how far they deviate.

Ten diverse structures are evaluated: eight molecules (H2O, CH4, NH3, C2H4, C2H2, SO2,
CH3OH and C6H6) and two periodic systems (diamond and graphene), matching the
translational symmetry test. Diamond is rattled with a fixed seed so that the atoms
carry nonzero forces: a perfect crystal's forces vanish by symmetry, which would leave
it probing energies only. Each structure is evaluated before and after each of 114
rigid rotations about the origin, forming a single cumulative walk through
orientation space: 14 steps increasing from 1 to 40°, followed by 100 uniformly random
steps, all generated with a fixed seed so that every model sees the identical walk. For the periodic structures the cell is rotated together with
the positions, preserving fractional coordinates: a rotation of the positions alone is
not a symmetry of a periodic crystal.
Random rotations sample orientation space rather than trusting one hand-picked axis
and angle, and generically avoid the special rotations (90 or 180° about a coordinate
axis) that permute or negate coordinates exactly in floating point and can cancel real
violations out of the comparison. Two uniformly random orientations are almost never
within 40° of each other, so the first 14 steps supply successive pairs at small and
intermediate relative angles: a smooth model must fail gently at small relative
angles, and an abrupt small-angle violation signals an implementation artifact rather
than learned approximate symmetry.

.. note::

    Forces are compared after rotating those of each orientation back into a common
    frame, where a perfect model gives identical forces for every orientation. The
    back-rotated forces are computed and stored during the calculation, so that the
    comparison is unaffected by the precision the output files are written at.

Metrics
-------

Successive orientations (the unrotated structure, then each rotation in turn) are
compared pairwise, giving 114 orientation pairs per structure; each metric is computed
over all pairs of all ten structures. Comparing each orientation with the next, rather
than every orientation with a fixed reference, means no single orientation is
privileged. A model that fails to complete the calculation for any structure is not
scored on the remainder: the metrics are left blank. Per-model plots resolve the
individual pairs, showing each structure's violation against the relative rotation
angle within the pair.

1. Mean ΔE

Mean absolute change in energy per atom between successive orientations.

2. Max ΔE

Worst-case absolute change in energy per atom between successive orientations.

3. Mean ΔF

Mean of the per-pair force MAE: the mean absolute difference between the two
orientations' force predictions, compared component-wise in a common frame, adapted
from the force rotational equivariance metric of MLIP Arena [1].

4. Max ΔF

Worst-case per-pair force MAE, as above.

[1] Chiang, Y., et al. "MLIP Arena: Advancing Fairness and Transparency in Machine
Learning Interatomic Potentials via an Open, Accessible Benchmark Platform." arXiv
preprint arXiv:2509.20630 (2025).

Computational cost
------------------

Medium: single-point evaluations only (10 structures evaluated in 115
orientations each), taking minutes per model on CPU for most models, up to tens
of minutes for the slowest.

Data availability
-----------------

None required; all structures are generated in ASE.
