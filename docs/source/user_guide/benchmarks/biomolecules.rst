============
Biomolecules
============

Protein folding stability
=========================

Summary
-------

Performance in keeping small proteins folded during molecular dynamics. For each
protein, an energy minimisation is run from the native (folded) reference
conformation, an NVT molecular dynamics simulation at 300 K is seeded with the
minimised coordinates, and the ability of the model to retain the fold is measured
along the trajectory. If the energy minimisation of a protein fails or is unstable,
the MD is instead run from the unminimised input structure.

The benchmark uses a small set of well-characterised proteins
(chignolin, tryptophan cage, and a capped villin headpiece) with experimental
reference structures. Each protein is solvated in a water box, sized to
accommodate the protein, giving systems of 1068 (chignolin), 2075 (tryptophan
cage), and 3441 (villin headpiece) atoms.

Metrics
-------

1. RMSD (chignolin)
2. RMSD (trp-cage)
3. RMSD (villin)

For each protein, the root mean square deviation (RMSD) of the C-alpha atoms from
the native reference structure is computed for each frame of the trajectory, then
averaged over the trajectory. A lower RMSD indicates the fold is retained. Each
protein is reported as a separate, equally weighted metric, so a model that fails
to simulate any of the proteins has no overall score.

For each protein, a line plot shows the RMSD from the reference structure along the
trajectory for each model.

Benchmark speed
------------------

Multi-day: one MD simulation per protein. Likely to take several days to run on GPU.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. Native reference structures taken from the
  Protein Data Bank (chignolin 1UAO, tryptophan cage 2JOF, villin headpiece).

Reference data:

* Experimental reference structures (X-ray and NMR) from the Protein Data Bank.

Protein sampling
================

Summary
-------

Performance in exploring the conformational space of small proteins during molecular
dynamics. A molecular dynamics simulation is run for each of a set of small
proteins (chignolin, Trp-cage, and a capped villin headpiece), and the sampled backbone
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
types and over the stable systems, and lower values are better. If two simulations fail,
the reported metrics describe only the remaining protein. Select a metric cell to
see its values by residue type for each model. Select a point to view the
starting structure of any protein contributing to that value.

Computational cost
------------------

Allow several days per model on a GPU as a rough estimate. Sampling uses the
same three solvated systems and MD protocol as the folding stability benchmark.
Wall time depends on the model and GPU.

Data availability
-----------------

Input structures:

* The same solvated chignolin (1UAO), Trp-cage (2JOF), and capped villin
  headpiece structures used for protein folding stability.

Reference data:

* MLIP Audit supplies residue-specific backbone and side-chain reference angles.
  The three distribution metrics use the angles in
  ``backbone_reference_data.json``. The `MLIP Audit sampling guide
  <https://instadeepai.github.io/mlipaudit/benchmarks/biomolecules/sampling.html>`_
  cites Lovell et al., *The Penultimate Rotamer Library* (2000).
