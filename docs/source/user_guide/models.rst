======
Models
======

The machine-learned interatomic potentials (MLIPs) known to ML-PEG are listed
below, with the configuration defined in ``ml_peg/models/models.yml``. Models
marked *(not currently enabled)* are commented out in ``models.yml`` and are not
run in the current benchmarks; they are listed here for completeness.

.. note::

   Citations and longer descriptions are still to be added for each model.

MACE
====

mace-mp-0a
----------

.. code-block:: yaml

   mace-mp-0a:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium"

mace-mp-0b3
-----------

.. code-block:: yaml

   mace-mp-0b3:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-0b3"

mace-mpa-0
----------

.. code-block:: yaml

   mace-mpa-0:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-mpa-0"

mace-omat-0
-----------

.. code-block:: yaml

   mace-omat-0:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-omat-0"

mace-matpes-r2scan
------------------

.. code-block:: yaml

   mace-matpes-r2scan:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: r2SCAN
     kwargs:
       model: "mace-matpes-r2scan-0"
       head: "matpes_pbe"
     dispersion_kwargs:
       xc: r2scan
       label: D4

mace-mh-1-omat
--------------


.. code-block:: yaml

   mace-mh-1-omat:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "mh-1"
       head: omat_pbe

MACE-OFF23(L)
-------------


.. code-block:: yaml

   MACE-OFF23(L):
     module: mace.calculators
     class_name: mace_off
     device: "auto"
     trained_on_dispersion: true
     kwargs:
       name: large

mace-omol
---------


.. code-block:: yaml

   mace-omol:
     module: mace.calculators
     class_name: mace_omol
     device: "auto"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model: "extra_large"

mace-mh-1-omol
--------------


.. code-block:: yaml

   mace-mh-1-omol:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model: "mh-1"
       head: omol

mace-polar-1-s
--------------


.. code-block:: yaml

   mace-polar-1-s:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-s"

mace-polar-1-m
--------------


.. code-block:: yaml

   mace-polar-1-m:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-m"

mace-polar-1-l
--------------


.. code-block:: yaml

   mace-polar-1-l:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-l"

Orb
===

orb-v3-consv-inf-omat
---------------------

.. code-block:: yaml

   orb-v3-consv-inf-omat:
     module: orb_models.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_conservative_inf_omat"

orb-v3-direct-inf-omat
----------------------

.. code-block:: yaml

   orb-v3-direct-inf-omat:
     module: orb_models.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_direct_inf_omat"

orb-v3-consv-omol
-----------------

.. code-block:: yaml

   orb-v3-consv-omol:
     module: orb_models.forcefield.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_conservative_omol"

PET
===

pet-mad
-------

.. code-block:: yaml

   pet-mad:
     module: pet_mad.calculator
     class_name: PETMADCalculator
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBEsol
     kwargs:
       version: "v1.0.2"
     dispersion_kwargs:
       xc: pbesol

UMA (FairChem)
==============

uma-s-1p1-omat
--------------


.. code-block:: yaml

   uma-s-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     level_of_theory: PBE
     trained_on_dispersion: false
     kwargs:
       model_name: "uma-s-1p1"
       task_name: "omat"

uma-m-1p1-omat
--------------


.. code-block:: yaml

   uma-m-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     level_of_theory: PBE
     trained_on_dispersion: false
     kwargs:
       model_name: "uma-m-1p1"
       task_name: "omat"

uma-s-1p1-omol
--------------


.. code-block:: yaml

   uma-s-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-s-1p1"
       task_name: "omol"

uma-s-1p2-omol
--------------


.. code-block:: yaml

   uma-s-1p2-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-s-1p2"
       task_name: "omol"

uma-m-1p1-omol
--------------


.. code-block:: yaml

   uma-m-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-m-1p1"
       task_name: "omol"

MatterSim
=========

mattersim-5M
------------


.. code-block:: yaml

   mattersim-5M:
     module: mattersim.forcefield
     class_name: MatterSimCalculator
     device: "cpu"
     load_path: "mattersim-v1.0.0-5m"
     trained_on_dispersion: false
     level_of_theory: PBE

GRACE
=====

GRACE-2L-OAM
------------


.. code-block:: yaml

   GRACE-2L-OAM:
     module: tensorpotential.calculator
     class_name: TPCalculator
     device: "cpu"
     trained_on_dispersion: false
     kwargs:
       model: "<local path to the GRACE-2L-OAM model>"
