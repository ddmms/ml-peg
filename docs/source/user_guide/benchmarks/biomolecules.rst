============
Biomolecules
============

Protein folding stability
=========================

Summary
-------

Performance in keeping small proteins folded during molecular dynamics. For each
protein, an NVT molecular dynamics simulation is run at 300 K starting from the
native (folded) reference conformation, and the ability of the model to retain the
fold is measured along the trajectory. The benchmark uses a small set of well
characterised proteins (chignolin, tryptophan cage, and an orexin/hypocretin
fragment) with experimental reference structures.

Metrics
-------

1. RMSD

The root mean square deviation of the C-alpha atoms from the native reference
structure is computed for each frame of the trajectory, then averaged over the
trajectory and across all proteins. A lower RMSD indicates the fold is retained.

2. TM score

The TM score of each trajectory frame against the native reference structure is
computed, then averaged over the trajectory and across all proteins. A TM score
closer to 1 indicates that the global fold is preserved.

3. Radius of gyration deviation

The maximum absolute deviation of the radius of gyration from the initial folded
state along the trajectory, taken across all proteins. A lower value indicates the
protein does not unfold or collapse.

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
  Protein Data Bank (chignolin 1UAO, tryptophan cage 2JOF, orexin-B 1CQ0).

Reference data:

* Experimental reference structures (X-ray and NMR) from the Protein Data Bank.
