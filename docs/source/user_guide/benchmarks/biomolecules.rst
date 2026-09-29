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
along the trajectory. The benchmark uses a small set of well-characterised proteins
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

Computational cost
------------------

Very high: one MD simulation per protein. Likely to take several days to run on GPU.

Data availability
-----------------

Input structures:

* MLIP Audit benchmark suite, InstaDeep. Native reference structures taken from the
  Protein Data Bank (chignolin 1UAO, tryptophan cage 2JOF, villin headpiece).

Reference data:

* Experimental reference structures (X-ray and NMR) from the Protein Data Bank.
