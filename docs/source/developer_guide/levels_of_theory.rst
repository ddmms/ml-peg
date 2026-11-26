================
Levels of theory
================

The level of theory describes the method used to generate a dataset (training or benchmarking),
for example, a DFT functional, post-Hartree-Fock method or experimental measurement. The ``level_of_theory``
field appears in two places in ML-PEG:

- ``models.yml`` — the functional or method of the dataset the MLIP was trained on (or of the head being used).
- ``metrics.yml`` — the reference method used to generate the benchmark data for a given metric.

Flagging level of theory mismatches
-----------------------------------

When a model's ``level_of_theory`` does not match the ``level_of_theory`` of **any** metric within
a benchmark, the app displays a coloured warning triangle in the model's table cell. These warnings
are designed to draw attention to any potential mismatch so that results can be interpreted in the
correct context. There are three warning types, colour-coded by the nature of the mismatch:

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

The app compares the base method and dispersion separately, including the
benchmark's configured D3 correction. A dispersion mismatch uses the DFT warning
icon. Damping is not compared.

.. note::

    A warning does not indicate that a model is unsuitable, but it is a prompt to consider whether the
    level-of-theory difference is relevant for your use case.

.. admonition:: Checking your assigned level of theory is correct

    Load up the app and inspect the benchmark tables in each category (summary tables don't show flags).
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

Record the base functional and dispersion method separately in both ``models.yml``
and ``metrics.yml``:

.. code-block:: yaml

   level_of_theory: PBE
   dispersion_method: D3

Accepted methods are ``D3``, ``D4``, ``VV10``, and ``MBD``.
Use ``none`` for no dispersion. Omission means unknown, except for models with
``trained_on_dispersion: false``, where it defaults to ``none``.
Existing labels such as ``PBE+D3`` are also supported. Keep named composite
methods (e.g. ``B97-3c``) and refitted functionals (e.g. ``wB97X-D3``)
intact, and record their dispersion method separately.

For models, these fields describe the training data. Analysis uses
``build_dispersion_name_map`` to identify models receiving the benchmark's D3
correction, independently of the display suffix. Regenerate analysis tables from
existing outputs to update this metadata. Calculations do not need to be rerun.
See :doc:`Adding models </developer_guide/add_models>` for runtime settings.

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
     - No level-of-theory comparison is made; no warning badge will be shown for this metric. Seen in physicality benchmarks where a metric is e.g. number of energy minima.


Reference table
---------------

The following strings are currently in use across the codebase. If your string is not listed,
please add it to this table when making a pull request for your benchmark or model.

**Models** (``ml_peg/models/models.yml``)

- ``B97-3c``
- ``PBE``
- ``PBE+D3``
- ``PBEsol``
- ``r2SCAN``
- ``ωB97M/def2-TZVPP``
- ``ωB97M``
- ``ωB97M/def2-TZVPD``
- ``ωB97M/def2-TZVPP``
- ``ωB97M/def2-TZVPPD``

**Benchmarks** (``ml_peg/analysis/<category>/<benchmark>/metrics.yml``)

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
- ``ωB97M + 1b CCSD(T)``
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
