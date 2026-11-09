==========
MAD-2 test
==========

Errors against six DFT references
=================================

Summary
-------

We compare pretrained models on 1948 structures from the MAD-2 test set. The six
references are the original MAD-2 r2SCAN labels and five VASP recalculations: MatPES
PBE, MatPES r²SCAN, OMat24, MPtrj (MPRelaxSet-based), and sAlex (Alexandria PBE). For
molecules and clusters, we used periodic boxes with 25 Å of vacuum. For surfaces and 2D
layers, we kept the in-plane cell and used 25 Å of vacuum along the slab normal. We
applied dipole corrections in both cases.

Metrics
-------

The tables report MAE, RMSE, the 95th percentile of absolute error and the largest
absolute error. Forces are in meV/Å, energies in meV/atom and stress in meV/Å³. Stress
is evaluated only for bulk structures. Energy errors are shown only where the model's
training energies are comparable to the reference. In "Available frames", each model is
evaluated on the frames it predicted. In "Common frames", models must cover at least 90%
of the selection to qualify. All qualifying models are then evaluated on exactly the
same frames.

Benchmark speed
---------------

Most models take a few minutes on a GPU, the slowest take up to an hour.

Data availability
-----------------

The input archive, ``mad-2-bench.zip``, contains the structures and reference labels. We
selected up to 100 structures per subset by farthest-point sampling. Each VASP file
contains only the structures that converged with that reference.

VASP settings follow
`MatPES <https://arxiv.org/abs/2503.04070>`_,
`OMat24 <https://arxiv.org/abs/2410.12771>`_ (including sAlex),
`PET-MAD <https://doi.org/10.1038/s41467-025-65662-7>`_ and
`Alexandria <https://doi.org/10.1002/adma.202210788>`_.
The `MAD-1.6 paper <https://arxiv.org/abs/2603.02089>`_ describes the MAD-2 dataset.
