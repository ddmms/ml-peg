========
Clusters
========

Cluster Forces
==============

Summary
-------

Performance in predicting forces on fixed geometries of neutral atomistic clusters
containing three to eight atoms.

Metrics
-------

1. Force MAE by cluster size

Mean absolute error (MAE) of force components across all atoms and clusters,
reported separately for clusters containing three, four, five, six, seven, and eight
atoms.

Each model is compared to the reference force array most appropriate for its training
domain. Models trained on broad atomistic datasets, or primarily on materials, are
compared to ``mad2_forces``. Models focused on organic chemistry are compared to
``omol25_forces``.

For OMOL25-routed calculations, all clusters are evaluated with ``charge=0`` and
``spin_multiplicity=1`` for clusters with an even number of electrons or
``spin_multiplicity=2`` for clusters with an odd number of electrons.

Computational cost
------------------

High: tests evaluate 60,000 clusters per model and are likely to take hours on CPU.

Data availability
-----------------

Input/reference data:

* Cluster structures and reference forces are distributed as a separate zip archive and
  downloaded on-demand from the ML-PEG data store. The calculation script uses the
  public ML-PEG S3 bucket to retrieve these inputs.
* The extended XYZ input contains both ``mad2_forces`` and ``omol25_forces`` reference
  arrays. Structures with non-finite values in the routed reference array did not
  converge at that level of theory and they are therefore excluded from the metric.
