========
Clusters
========

Cluster Forces
==============

Summary
-------

Performance in predicting forces on neutral atomistic clusters containing three to eight
atoms where the chemical elements are chosen randomly. This benchmark is designed to
test the universality and generalizability of the interatomic potentials.

Metrics
-------

1. MAD2 force MAE by cluster size

Mean absolute error (MAE) of force components across all atoms and clusters,
reported separately for clusters containing three, four, five, six, seven, and eight
atoms. Each model is compared to the ``mad2_forces`` reference values.

2. OMol25 force MAE by cluster size

Mean absolute error (MAE) of force components across all atoms and clusters,
reported separately for clusters containing three, four, five, six, seven, and eight
atoms. Each model is compared to the ``omol25_forces`` reference values.

For all calculations, clusters are evaluated with ``charge=0`` and either
``spin_multiplicity=1`` for clusters with an even number of electrons or
``spin_multiplicity=2`` for clusters with an odd number of electrons.

Computational cost
------------------

Medium: tests evaluate 60,000 small clusters per model and are expected to take less
than an hour locally.

Data availability
-----------------

Cluster structures and reference forces are distributed as a separate zip archive and
downloaded on-demand from the ML-PEG data store. The calculation script uses the
public ML-PEG S3 bucket to retrieve these inputs.

The structures are randomly generated neutral clusters ("n-mers") containing three
to eight atoms. There are 10,000 clusters for each size from three to eight atoms,
inclusive. Atoms were placed randomly while enforcing all pairwise interatomic distances
to lie between 2.0 and 2.5 Å; element identities were sampled randomly from the 18
elements of the first three rows of the periodic table.
  
The extended XYZ input contains both ``mad2_forces`` and ``omol25_forces`` reference
arrays. Non-finite reference or predicted force components are excluded from the
evaluation of the corresponding metric. These correspond to non-converged DFT
calculations and/or model failures in the prediction (for example, due to
unsupported chemical elements).

Reference labels:

* ``mad2_forces``: r2SCAN reference forces calculated with FHI-AIMS, using settings
  consistent with the MAD-1.5 dataset.
* ``omol25_forces``: ``ωB97M-V/def2-TZVPD`` reference forces calculated with ORCA,
  using settings consistent with the OMol25 dataset. OMol-style DFT calculations used
  total charge zero and spin multiplicities of 1 for even-electron clusters or 2 for
  odd-electron clusters.

In both cases, energies and forces were set to ``NaN`` in the rare cases where the DFT
calculation did not converge.

Further details on the motivation for n-mer benchmarks of universal MLIPs are
available at https://doi.org/10.1063/5.0303302.
