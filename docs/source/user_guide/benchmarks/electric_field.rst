==============
Electric Field
==============

Energy response
===============

Summary
-------

Performance in predicting the change in total energy of a molecule when a static,
homogeneous external electric field is applied. Two families of linear organic
molecules are included, 20 systems each: linear alkanes
(:math:`\mathrm{C}_n\mathrm{H}_{2n+2}`) and cumulenes
(:math:`\mathrm{C}_n\mathrm{H}_4`), both for even chain lengths
:math:`n = 2, 4, \ldots, 40`.

Every molecule appears twice in the dataset: once at zero field, and once with a field
of 0.0514 V/Å (:math:`10^{-3}` a.u., :math:`5.14 \times 10^8` V/m) applied along the
chain axis. Geometries are identical between the two, so the energy response measures
the electronic response to the field alone, with no field-induced relaxation.

The two families probe very different polarisabilities. Alkanes are saturated, and
their response grows slowly and roughly linearly with chain length, from 0.0004 eV at
:math:`\mathrm{C_2H_6}` to 0.0096 eV at :math:`\mathrm{C_{40}H_{82}}`. Cumulenes carry a
delocalised :math:`\pi` system along the chain, and their response grows far faster,
from 0.0005 eV at :math:`\mathrm{C_2H_4}` to 0.28 eV at :math:`\mathrm{C_{40}H_4}` — a
factor of 30 larger than the longest alkane.

.. note::

    A model that takes no electric field as input predicts a response of exactly zero
    for every system, because the two structures of each pair differ only in their
    recorded field. This benchmark therefore discriminates only between field-aware
    models; for all other models it reports the magnitude of the reference response
    itself.

Metrics
-------

**MAE**

For each molecule, the energy response is the magnitude of the energy difference
between the field and zero-field structures:

.. math::
    \Delta E = \left| E(\mathbf{F}) - E(\mathbf{0}) \right|

The mean absolute error of :math:`\Delta E` against the reference responses is reported
in three columns: ``Alkanes MAE`` and ``Cumulenes MAE`` over each family separately,
and ``Total MAE`` over all 40 systems.

For all three columns the ``good`` threshold is 0.01 eV and the ``bad`` threshold is
5.0 eV. Each MAE is converted to a normalised score in :math:`[0, 1]` by linear
interpolation between ``good`` (score = 1) and ``bad`` (score = 0). The overall
``Score`` is the weighted average of the per-family scores: ``Alkanes MAE`` and
``Cumulenes MAE`` carry weight 1, and ``Total MAE`` is reported for information only,
with weight 0.

Computational cost
------------------

Low: 80 single-point energy calculations, with no geometry optimisation. The largest
system is :math:`\mathrm{C_{40}H_{82}}` at 122 atoms.

Data availability
-----------------

Input structures:

* Idealised linear alkane and cumulene geometries generated for this benchmark.

Reference data:

* Computed for this benchmark with `ORCA <https://www.faccts.de/orca/>`_ at the
  :math:`\omega\text{B97M-V/def2-TZVPD}` level of theory: a range-separated hybrid
  meta-GGA functional with non-local VV10 correlation, in a triply polarised,
  diffuse-augmented basis.
