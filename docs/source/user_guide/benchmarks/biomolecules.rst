============
Biomolecules
============

Protein sampling
================

Summary
-------

Performance in exploring the conformational space of small proteins during molecular
dynamics. A short molecular dynamics simulation is run for each of a set of small
proteins (chignolin, Trp-cage, and an orexin beta fragment), and the sampled backbone
dihedral angles are compared to reference distributions. This probes whether a model
samples physically reasonable protein conformations rather than only reproducing
static reference geometries.

Metrics
-------

1. Backbone Dihedral RMSD
2. Backbone Hellinger Distance
3. Backbone Outliers Ratio

For each system, the sampled backbone (phi/psi) dihedral angles are collected across the
trajectory and binned into a distribution per residue type. This distribution is
compared to a reference distribution using the root mean square deviation (RMSD) and the
Hellinger distance. The outliers ratio measures the fraction of sampled dihedrals that
lie far from any point in the reference data. All three metrics are averaged over residue
types and over the stable systems, and lower values are better.

Computational cost
------------------

High: each model runs several molecular dynamics simulations of solvated proteins.

Data availability
-----------------

Input structures:

* Experimental structures from the Protein Data Bank (chignolin 1UAO, Trp-cage 2JOF,
  orexin beta 1CQ0).

Reference data:

* Reference backbone and side-chain dihedral distributions derived from molecular
  dynamics reference simulations.
