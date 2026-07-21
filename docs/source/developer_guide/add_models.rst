=============
Adding models
=============

ML-PEG gets its model list from ``ml_peg/models/models.yml`` by default. Each
top-level key is the model name used in calculation outputs, analysis, and the
app. Calculation scripts load these entries with ``load_models`` and then call
``model.get_calculator(...)`` to obtain an ASE calculator.

You can either edit the default registry or keep local/private entries in a
separate YAML file and pass it on the command line:

.. code-block:: bash

   ml_peg calc --models-file my_models.yml --models my-mace-model
   ml_peg analyse --models-file my_models.yml --models my-mace-model
   ml_peg app --models-file my_models.yml --models my-mace-model

To check which model names ML-PEG can see, use:

.. code-block:: bash

   ml_peg list models
   ml_peg list models --models-file my_models.yml

Model entry format
------------------

A typical entry has this shape:

.. code-block:: yaml

   model-name:
     module: package.module
     class_name: CalculatorFactoryOrClass
     device: cuda
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       option_name: option_value
     dispersion_kwargs:
       option_name: option_value

The common fields are:

``module``
   Python module containing the calculator factory or class.

``class_name``
   Name imported from ``module``.

``device``
   Device passed to calculators that support it. Options: ``cuda``, ``cpu`` or ``auto``.

``trained_on_dispersion``
   Whether the model's training data already included dispersion corrections.
   Some benchmark scripts call ``add_d3_calculator``. If this field is
   ``false``, ML-PEG may add a separate D3 correction for those benchmarks; if it
   is ``true``, the base calculator is used unchanged.

``level_of_theory``
   Functional or method represented by the model training data. The app compares
   this string against benchmark metric metadata and displays warnings when they
   differ. See :doc:`Levels of theory </developer_guide/levels_of_theory>` for
   naming conventions.

``datasets``
   Optional list of training-dataset names, e.g. ``[MPtrj]`` or
   ``[MPtrj, sAlex]``. Each name maps, via
   ``ml_peg/app/data/element_coverage.json``, to the elements that dataset
   covers. A model's coverage is the **union** across all listed datasets, so the
   app's element filter can keep or exclude elements by a model's coverage. Names
   are **case-sensitive** and every one must be a key in ``element_coverage.json``
   (each entry has a ``supported`` element list and a ``number`` count); add a
   new dataset entry there if the model's training set is not yet listed. Set
   ``datasets: null`` when the model was trained on e.g. a non-public dataset (then
   express its coverage entirely through ``additional_supported_elements``).

``additional_supported_elements``
   Optional list of element symbols the model supports *beyond* its ``datasets``
   coverage. A model's total coverage is the union of its datasets' elements and
   this list. To determine coverage empirically, run
   ``ml_peg/models/element_coverage/find_supported_elements.py`` (once per ``uv sync --extra <backend>``,
   since model backends conflict), then
   ``ml_peg/models/element_coverage/compare_supported_elements.py`` to see which ``datasets`` tags fit
   and which elements to record here as extras.

``kwargs``
   Keyword arguments forwarded to the calculator constructor or factory. Here you can
   input kwargs you would usually use for the calculator.

``dispersion_kwargs``
   Optional settings used by ``add_d3_calculator`` when a benchmark adds a
   separate dispersion correction. These are option/value pairs passed through
   to the dispersion wrapper, so the accepted keys depend on that wrapper.

``overwrite_dtype``
   Optional precision override for model wrappers that support it. Most
   benchmarks request either ``precision="high"`` or ``precision="low"``; this
   field forces a specific dtype regardless of that request.

Examples
-------------

MACE-MP foundation model:

.. code-block:: yaml

   mace-mp-0a:
     module: mace.calculators
     class_name: mace_mp
     device: cuda
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: medium

MACE checkpoint from a local path:

.. code-block:: yaml

   my-mace-model:
     module: mace.calculators
     class_name: mace_mp
     device: cuda
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: /absolute/path/to/my_mace_checkpoint.model

Models with a specific head or task can pass that head in ``kwargs``:

.. code-block:: yaml

   mace-mh-1-omol:
   module: mace.calculators
   class_name: mace_mp
   device: "cuda"
   trained_on_dispersion: true
   level_of_theory: ωB97M-V/def2-TZVPD
   kwargs:
      model: "mh-1"
      head: omol

   uma-s-1p1-omol:
   module: fairchem.core
   class_name: FAIRChemCalculator
   device: "cuda"
   trained_on_dispersion: true
   level_of_theory: ωB97M-V/def2-TZVPD
   kwargs:
      model_name: "uma-s-1p1"
      task_name: "omol"


Supported model families
------------------------------

MACE entries use the generic ASE calculator wrapper. Common class names include
``mace_mp``, ``mace_off``, ``mace_omol`` and ``mace_polar``.

MACE-Polar foundation model:

.. code-block:: yaml

   mace-polar-1-m:
     module: mace.calculators
     class_name: mace_polar
     device: cuda
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: polar-1-m

ORB entries use the dedicated ``OrbCalc`` wrapper:

.. code-block:: yaml

   orb-v3-consv-inf-omat:
     module: orb_models.inference.calculator
     class_name: OrbCalc
     device: cuda
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       name: orb_v3_conservative_inf_omat

FairChem/UMA entries use the dedicated ``FAIRChemCalculator`` branch:

.. code-block:: yaml

   uma-s-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: cuda
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model_name: uma-s-1p1
       task_name: omat

PET-MAD entries use the dedicated ``PETMADCalculator`` branch:

.. code-block:: yaml

   pet-mad:
     module: pet_mad.calculator
     class_name: PETMADCalculator
     device: cuda
     trained_on_dispersion: false
     level_of_theory: PBEsol
     kwargs:
       version: v1.0.2
     dispersion_kwargs:
       xc: pbesol

Other ASE-compatible MLIP calculators can usually be added by specifying their
``module``, ``class_name`` and constructor ``kwargs``. That is enough when the
calculator accepts the same common arguments as the generic ML-PEG model wrapper.

For a completely new model family, or for a calculator that needs special setup
logic, add a small wrapper class in ``ml_peg/models/models.py`` and route to it
from ``load_models`` in ``ml_peg/models/get_models.py``. In those cases the YAML
entry records the model configuration, while the Python wrapper defines how
ML-PEG constructs the calculator.

Checking a new entry
--------------------

After adding a model, run a small calculation first:

.. code-block:: bash

   ml_peg list models --models-file my_models.yml
   ml_peg calc --category molecular_crystal --test X23 --models-file my_models.yml --models my-mace-model

If loading fails, check that the optional dependency is installed, the
``module``/``class_name`` pair can be imported, local checkpoint paths are valid,
and the model's ``kwargs`` match the calculator API.
