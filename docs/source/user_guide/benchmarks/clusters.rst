========
Clusters
========

Both benchmarks in this category use the same set of randomly generated neutral
clusters ("n-mers") containing three to eight atoms, and differ only in the reference
forces they are compared to. Clusters of this kind test how well universal
interatomic potentials perform on highly diverse local environments.
Further details on the motivation for n-mer benchmarks of universal MLIPs are
available at https://doi.org/10.1063/5.0303302.

.. _cluster-forces-data:

The benchmark data consists of 10,000 clusters for each size from three to eight atoms,
inclusive. Element identities were sampled randomly from the 15 elements of the first three rows of the
periodic table, excluding the noble gases (H, Li–F, and Na–Cl). Atoms were placed
randomly so that no two atoms are closer than 2.0 Å, and the nearest neighbour of
every atom is between 2.0 and 2.5 Å away.

All clusters are neutral. The r2SCAN reference calculations are spin-unpolarised. The
ωB97M-V reference calculations use spin multiplicity 1 for clusters with an even
number of electrons, and 2 for clusters with an odd number of electrons. In both
benchmarks, this charge and spin multiplicity are passed to models that accept them.

Clusters are excluded from a benchmark when the corresponding reference calculation
did not converge: 175 clusters for the r2SCAN references, and 4,033 clusters for the
ωB97M-V references. If a model fails to predict forces for any of the remaining
clusters of a given size, for example due to unsupported elements, its MAE for that
size is not reported.

.. _cluster-forces-thresholds:

Both benchmarks use the same thresholds for each cluster size. The "good" threshold is
the force MAE between the r2SCAN and ωB97M-V reference forces, rounded up to three
decimal places, so that a model exactly reproducing either level of theory scores 1 on
both benchmarks. Adding D3 to the r2SCAN forces changes this MAE by less than 0.001
eV/Å. The "bad" threshold is half the MAE of a model predicting zero forces, averaged
over the two references.

.. list-table::
   :header-rows: 1

   * - Cluster size
     - 3
     - 4
     - 5
     - 6
     - 7
     - 8
   * - Good / (eV/Å)
     - 0.232
     - 0.283
     - 0.299
     - 0.315
     - 0.319
     - 0.325
   * - Bad / (eV/Å)
     - 0.547
     - 0.616
     - 0.633
     - 0.642
     - 0.658
     - 0.679


.. _cluster-forces-r2scan:

Cluster Forces r2SCAN
=====================

Summary
-------

Performance in predicting forces on random neutral clusters of three to eight atoms,
compared to r2SCAN reference forces. The reference forces were calculated with
FHI-aims, using settings consistent with the MAD-1.5 dataset.

Metrics
-------

1. Force MAE by cluster size

Mean absolute error (MAE) of force components across all atoms and clusters,
reported separately for clusters containing three, four, five, six, seven, and eight
atoms. Thresholds are described at the
:ref:`top of this page <cluster-forces-thresholds>`.

Models are evaluated using their native predictions, without adding a dispersion
correction, since the r2SCAN reference forces do not include one. Differences
between a model's training level of theory and r2SCAN therefore contribute to the
reported errors.

Benchmark speed
---------------

Medium: these tests evaluate 60,000 small clusters per model. They are expected to take
on the order of minutes on a GPU.

Data availability
-----------------

Input structures:

* Randomly generated clusters, described at the
  :ref:`top of this page <cluster-forces-data>`.

Reference data:

* r2SCAN forces calculated with FHI-aims, using settings consistent with the MAD-1.5
  dataset.

* C. Malosso, F. Bigi, P. Pegolo, J. W. Abbott, P. Loche, M. Rossi, T. J. Goncalves,
  S. De, M. Ceriotti, and A. Mazitov, High-quality, high-information datasets for
  universal atomistic machine learning, arXiv:2603.02089 (2026).


.. _cluster-forces-wb97mv:

Cluster Forces ωB97M-V
======================

Summary
-------

Performance in predicting forces on random neutral clusters of three to eight atoms,
compared to ωB97M-V reference forces. The reference forces were calculated
with ORCA, using settings consistent with the OMol25 dataset.

Metrics
-------

1. Force MAE by cluster size

Mean absolute error (MAE) of force components across all atoms and clusters,
reported separately for clusters containing three, four, five, six, seven, and eight
atoms. Thresholds are described at the
:ref:`top of this page <cluster-forces-thresholds>`.

ωB97M-V includes dispersion through its VV10 non-local correlation term, so a D3(BJ)
dispersion correction is added to models that were not trained on
dispersion-corrected data. These models are labelled with a ``-D3`` suffix.

Benchmark speed
---------------

Medium: these tests evaluate 60,000 small clusters per model. They are expected to take
on the order of minutes on a GPU.

Data availability
-----------------

Input structures:

* Randomly generated clusters, described at the
  :ref:`top of this page <cluster-forces-data>`.

Reference data:

* ωB97M-V forces calculated with ORCA, using settings consistent with the
  OMol25 dataset.

* D. S. Levine et al., The Open Molecules 2025 (OMol25) Dataset, Evaluations, and
  Models, arXiv:2505.08762 (2025).
