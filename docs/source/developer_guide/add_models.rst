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

Overview of the process
-----------------------

Adding a model to ML-PEG usually involves the following steps. Only the first
and last are always required: a model from an already supported family, or any
ASE-compatible calculator that needs no special setup, is just a YAML entry.

1. **Add an entry to** ``models.yml`` (or your own models file) describing the
   model — see `Model entry format`_.
2. **Add the dependency** as an optional extra in ``pyproject.toml``, if the
   calculator comes from a package ML-PEG does not already depend on — see
   `Dependencies and extras`_.
3. **Add a calculator wrapper** in ``ml_peg/models/models.py`` and route to it
   from ``load_models`` in ``ml_peg/models/get_models.py``, if the calculator
   needs setup beyond ``module``/``class_name``/``kwargs`` — see
   `Adding a new model family`_.
4. **Record element coverage** via ``datasets`` and
   ``additional_supported_elements``, so the app's element filter works for the
   new model.
5. **Document the model** in ``docs/source/user_guide/models.rst``, which lists
   the YAML entry for every model shipped with ML-PEG.
6. **Check the entry loads and runs** — see `Checking a new entry`_.

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
   Name imported from ``module``. ``load_models`` also matches on this name to
   decide which ML-PEG wrapper to build, so it must match one of the ``case``
   branches in ``load_models`` for model families that need a dedicated wrapper.

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
   benchmarks request either ``precision="high"`` or ``precision="low"``, which
   each wrapper maps onto its backend's precision argument; this field replaces
   whatever that mapping would have produced, regardless of the request.

   The value is passed straight through to the backend, so its accepted form is
   whatever that backend expects — usually ``float32`` or ``float64``, but Orb
   takes its own precision strings and ``grace_fm`` takes a suffix appended to
   ``kwargs.model`` (``-fp64``) that selects a different published checkpoint.
   Check the relevant ``get_calculator`` in ``ml_peg/models/models.py`` before
   setting it; wrappers whose backend has no dtype argument ignore the field.

Examples
--------

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
     device: cuda
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model: mh-1
       head: omol

   uma-s-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: cuda
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: uma-s-1p1
       task_name: omol

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

PET-MAD entries use UPET's ``UPETCalculator``:

.. code-block:: yaml

   pet-mad:
     module: upet.calculator
     class_name: UPETCalculator
     device: cuda
     trained_on_dispersion: false
     level_of_theory: PBEsol
     kwargs:
       model: pet-mad-s
       version: 1.0.2
     dispersion_kwargs:
       xc: pbesol

DPA entries use the dedicated ``DpaCalc`` wrapper selected by DeePMD's ``DP``
class name and refer to checkpoints through DeePMD's packaged aliases:

.. code-block:: yaml

   dpa-3p3-1M-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OpenLAM-v1, OMAT, MPtrj, OC20, OC22, ODAC23, SPICE2]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA-3.3-1M
       head: Omat24

Here ``OpenLAM-v1`` is the umbrella pretraining collection, while familiar
constituents are also listed individually so their known domains and element
coverage remain visible. Its coverage entry is the verified union of its
constituent datasets, not the checkpoint's broader executable type map.

GRACE, SevenNet, MatterSim and Vivace entries similarly select their wrappers
through ``class_name`` (``grace_fm``, ``SevenNetCalculator``,
``MatterSimCalculator`` and ``MLFFCalculator`` respectively).

Other ASE-compatible MLIP calculators can usually be added by specifying their
``module``, ``class_name`` and constructor ``kwargs``. Any ``class_name`` that
does not match a dedicated branch falls through to the generic
``GenericASECalc`` wrapper, which imports ``class_name`` from ``module`` and
calls it with ``kwargs`` — enough when the calculator accepts the same common
arguments as the generic ML-PEG model wrapper.

Dependencies and extras
-----------------------

Model backends are optional dependencies, declared one extra per family in
``[project.optional-dependencies]`` in ``pyproject.toml``:

.. code-block:: toml

   [project.optional-dependencies]
   mynet = [
       "mynet == 1.2.3",
   ]

Pin the version, and add environment markers (e.g.
``; python_version >= '3.11'``) if the package does not support the full
``requires-python`` range. Backends with mutually incompatible requirements must
be declared in ``[tool.uv].conflicts`` so ``uv`` can resolve them into separate
environments:

.. code-block:: toml

   conflicts = [
       [
           { extra = "mynet" },
           { extra = "mace" },
       ],
   ]

Then regenerate the lockfile and install the new extra:

.. code-block:: bash

   uv lock
   uv sync --extra mynet --extra d3

Because conflicting extras cannot be installed together, a model's benchmarks
are typically run in a separate environment from other families.

Adding a new model family
-------------------------

For a completely new model family, or a calculator that needs special setup
logic (patching a dependency, mapping ``precision`` onto a non-standard dtype
argument, constructing a predictor before the calculator), add a wrapper in
``ml_peg/models/models.py``:

* Subclass ``GenericASECalc`` if the calculator can be imported and called as
  ``class_name(**kwargs)``, and you only need to adjust how precision is passed
  or run setup first (as ``UPETCalc`` and ``GraceCalc`` do).
* Subclass ``SumCalc`` directly if construction is bespoke (as ``OrbCalc``,
  ``FairChemCalc``, ``SevenNetCalc`` and ``VivaceCalc`` do). ``SumCalc``
  provides ``add_d3_calculator`` and the ``trained_on_dispersion`` /
  ``dispersion_kwargs`` fields that every model needs.

The wrapper must implement ``get_calculator(self, *, precision, **kwargs)``,
call ``check_precision(precision)``, map ``"low"``/``"high"`` onto the
calculator's dtype argument, let ``overwrite_dtype`` (stored as
``default_dtype``) override that mapping, and return a loaded ASE calculator.
Subclasses of ``GenericASECalc`` inherit ``default_dtype``, while ``SumCalc``
subclasses (e.g. ``OrbCalc``, ``FairChemCalc``) must declare it. Leave it out
only if the backend has no dtype argument to override, and note that in the
``get_calculator`` docstring.

Import the backend inside ``get_calculator`` rather than at module level, so
``models.py`` stays importable without the extra installed. Optionally define an
``available`` property that reports whether the backend can be loaded.

Then route to it from ``load_models`` in ``ml_peg/models/get_models.py`` by
adding a ``case`` for the ``class_name`` used in the YAML:

.. code-block:: python

   case "MyNetCalculator":
       loaded_models[name] = MyNetCalc(
           device=cfg.get("device", "cpu"),
           default_dtype=cfg.get("overwrite_dtype", None),
           kwargs=cfg.get("kwargs", {}),
           trained_on_dispersion=cfg.get("trained_on_dispersion", False),
           dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
       )

Pass every field the wrapper supports here — a field the wrapper accepts but
``load_models`` never forwards is ignored without any error.

The YAML entry records the model configuration, while the Python wrapper defines
how ML-PEG constructs the calculator.

Checking a new entry
--------------------

After adding a model, run a small calculation first:

.. code-block:: bash

   ml_peg list models --models-file my_models.yml
   ml_peg calc --category molecular_crystal --test X23 --models-file my_models.yml --models my-mace-model

For DPA models, install the backend and exercise a small representative OMat
system with:

.. code-block:: bash

   uv sync --extra dpa
   ml_peg calc --category <category> --test <test> --models dpa-4-nano-omat

Check the loaded checkpoint's actual parameter or graph dtype as well as finite
energies, forces, and stress.

Analysis for the new model can then be added to the existing tables and plots,
without rerunning analysis for all other models, using ``--update``:

.. code-block:: bash

   ml_peg analyse --category molecular_crystal --test X23 --models my-mace-model --update

Saved results are only preserved for models defined in the models file used by the
run, so ``my_models.yml`` must define every model to be shown, not only the new one.
Add the new model to the default ``models.yml``, as in the command above, or copy
the existing entries into ``my_models.yml`` and pass ``--models-file my_models.yml``.

See :doc:`Running tests </developer_guide/running>` for details.

If loading fails, check that the optional dependency is installed, the
``module``/``class_name`` pair can be imported, local checkpoint paths are valid,
and the model's ``kwargs`` match the calculator API.
