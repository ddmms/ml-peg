================
Levels of theory
================

The level of theory describes the method used to generate a dataset (for model training or benchmarking),
for example, a DFT functional, post-Hartree-Fock method or experimental measurement. The ``level_of_theory``
field appears in two places in ML-PEG:

- ``models.yml`` — the functional or method the MLIP was trained on.
- ``metrics.yml`` — the reference method used to generate the benchmark data for a given metric.

Flagging level of theory mismatches
-----------------------------------

When a model's ``level_of_theory`` does not match the ``level_of_theory`` of **any** metric within
a benchmark, the app displays a coloured warning triangle in the model's table cell. A warning can
appear even if some metrics in the benchmark do match as the warnings are designed to draw attention
to any potential mismatch so that results can be interpreted in the correct context. There are three
warning types, colour-coded by the nature of the mismatch:

.. list-table::
   :header-rows: 1
   :widths: 8 32 60

   * - Icon
     - Warning
     - Meaning
   * - |icon-dft|
     - DFT functional mismatch
     - The model and benchmark use different DFT functionals (e.g. model trained on PBE, benchmark
       computed with r2SCAN).
   * - |icon-high-level|
     - High-level theory mismatch
     - The benchmark uses a high-level reference method (e.g. CCSD(T), MP2, DMC) but the model
       was trained on DFT data.
   * - |icon-experimental|
     - Experimental reference mismatch
     - The benchmark uses experimental reference data.

The icon colour is currently determined by the mismatched benchmark metric's ``level_of_theory``, not the
model's. When mismatches of different types occur within the same benchmark, the highest-priority
warning is shown: DFT functional mismatch > High-level theory mismatch > Experimental reference mismatch.

.. note::

    A warning does not indicate that a model is unsuitable, but it is a prompt to consider whether the
    level-of-theory difference is relevant for your use case.

.. admonition:: Checking your assigned level of theory is correct

    Load up the app and inspect the benchmark tables in each category (summary tables don't show flags)
    For example, a PBE benchmark should have no warning for models trained on PBE datasets (e.g. MACE-MP-0a),
    but a DFT functional mismatch warning for models trained on an r2SCAN dataset (e.g. MACE-MATPES-r2SCAN).
    A model trained on PBE should have no warnings for the PBE phonon benchmark (Bulk Crystals),
    a "high-level theory mismatch" for the DLPNO-CCSD(T)/CBS Wiggle150 benchmark (Molecular Systems), and
    an "Experimental reference mismatch" for the experimental lattice constants metric (Bulk Crystals).


Naming conventions
------------------

.. warning::

    Before introducing a new ``level_of_theory`` string, check whether an equivalent string already
    exists in ``models.yml`` or any ``metrics.yml`` file (full list below). The app compares these strings
    **exactly** to decide whether to display a level-of-theory warning badge on a model's table cell.
    If the strings do not match, this could lead to an incorrect warning. When adding a new benchmark
    or model, it is recommended to check that you see the expected flags.

Standard methods
~~~~~~~~~~~~~~~~

For standard DFT functionals and post-Hartree-Fock methods, write the name as it would appear in a
paper:

.. code-block:: yaml

   level_of_theory: PBE
   level_of_theory: r2SCAN
   level_of_theory: CCSD(T)/CBS

Dispersion corrections
~~~~~~~~~~~~~~~~~~~~~~

Append the dispersion correction to the base functional using a ``+`` separator, with no spaces:

.. code-block:: yaml

   level_of_theory: PBE+D3

The same convention applies on both sides: if a model was trained on PBE+D3 data, its ``models.yml``
entry should read ``PBE+D3``, and any benchmark metric derived from PBE+D3 reference data should
also read ``PBE+D3``.

.. note::

    Dispersion corrections added at inference time (via ``trained_on_dispersion: false`` in
    ``models.yml``) are handled separately by the ``add_d3_calculator`` mechanism and do not affect
    the ``level_of_theory`` string. Only set ``level_of_theory`` to include ``+D3`` if the training
    data itself was generated with dispersion corrections.

.. _special-values:

Special values
~~~~~~~~~~~~~~

Some benchmark metrics use reference data that is experimental or does not correspond to a standard QM method.
The following special strings are used:

.. warning::

    These strings must be written exactly as shown (case sensitive).

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - String
     - Meaning
   * - ``Experimental``
     - Reference values taken from experiment (e.g. measured lattice constants, enthalpies).
   * - ``null``
     - No level-of-theory comparison is made; no warning badge will be shown for this metric. Seen in physicality benchmarks where a metric is e.g. number of energy minima


Reference table
---------------

The following strings are currently in use across the codebase. If your string is not listed,
please add it to this table when making a pull request for your benchmark or model.

**Models** (``models.yml``)

- ``PBE``
- ``PBEsol``
- ``r2SCAN``
- ``ωB97M-V``
- ``ωB97M-V/def2-TZVPD``

**Benchmarks** (``metrics.yml``)

- ``PBE``
- ``r2SCAN``
- ``r2SCAN-3c``
- ``CCSD(T)``
- ``CCSD(T)/CBS``
- ``CCSD(T)-F12/cc-pVDZ-F12``
- ``CCSDT(Q)/CBS``
- ``DLPNO-CCSD(T)/CBS``
- ``DLPNO-CCSD(T)/cc-pVTZ``
- ``PNO-LCCSD(T)-F12/AVQZ``
- ``DMC``
- ``ph-AFQMC``
- ``wb97md3 + 1b CCSD(T)``
- ``Experimental`` (see :ref:`special values <special-values>`)
- ``null`` (see :ref:`special values <special-values>`)


See also
--------

- :doc:`Adding benchmarks </developer_guide/add_benchmarks>` — where ``metrics.yml`` is configured.
- :doc:`Adding models </developer_guide/add_models>` — where ``models.yml`` is configured.
- :doc:`Benchmark scoring </developer_guide/scoring_and_normalisation>` — how metric configuration is used.


.. |icon-dft| raw:: html

   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="vertical-align:middle">
     <path fill="#d9534f" d="M8 1.1 15.2 14H0.8Z"/>
     <path fill="#212121" d="M7.25 11.9h1.5V13h-1.5zM7.36 5.2h1.28l.3 5.4h-1.88z"/>
   </svg>

.. |icon-high-level| raw:: html

   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="vertical-align:middle">
     <path fill="#f0ad4e" d="M8 1.1 15.2 14H0.8Z"/>
     <path fill="#212121" d="M7.25 11.9h1.5V13h-1.5zM7.36 5.2h1.28l.3 5.4h-1.88z"/>
   </svg>

.. |icon-experimental| raw:: html

   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="20" height="20" style="vertical-align:middle">
     <path fill="#5cb85c" d="M8 1.1 15.2 14H0.8Z"/>
     <path fill="#212121" d="M7.25 11.9h1.5V13h-1.5zM7.36 5.2h1.28l.3 5.4h-1.88z"/>
   </svg>
