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
along the trajectory. The benchmark uses a small set of well characterised proteins
(chignolin, tryptophan cage, and a capped, solvated villin headpiece) with
experimental reference structures.

Metrics
-------

1. RMSD

The root mean square deviation (RMSD) of the C-alpha atoms from the native reference
structure is computed for each frame of the trajectory, then averaged over the
trajectory and across all proteins. A lower RMSD indicates the fold is retained.

A line plot shows the RMSD from the reference structure along the trajectory,
averaged across the proteins, for each model.

Computational cost
------------------

High: one MD simulation per protein. Faster inference can be achieved using the
jax-accelerated simulations in MLIP Audit directly.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. Native reference structures taken from the
  Protein Data Bank (chignolin 1UAO, tryptophan cage 2JOF, villin headpiece).

Reference data:

* Experimental reference structures (X-ray and NMR) from the Protein Data Bank.
