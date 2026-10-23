================
Porous Materials
================


QMOF
====

Summary
-------

Performance in predicting energies per atom for 13905 metal organic frameworks (MOFs)
from the Quantum MOF Database (QMOF), a public online collection of electronic and
physical properties for more than 20,000 MOFs and coordination polymers[1].


Metrics
-------

MAE

Mean Absolute Error of energy per atom for all systems.

Benchmark speed
------------------

Medium: calculations are expected to take 10s of minutes on a A100 GPU per model.


Data availability
-----------------

Input data:

Structures were taken from the Quantum MOF Database (QMOF), with some structures
removed for compatibility with the MACE_MP paper due to missing elements or
inconsistent DFT.

[1] Rosen A, et al
**Machine learning the quantum-chemical properties of metal–organic frameworks for accelerated materials discovery**
Matter, 2021; 4, 1578-1597, 10.1016/j.matt.2021.02.015

[2] Batatia I, et al
A foundation model for atomistic materials chemistry
J. Chem. Phys. 163, 184110 (2025), https://doi.org/10.1063/5.0297006

Reference data:

* Same as input data
* PBE-D3


MOF adsorption enthalpy
=======================

Summary
-------

Performance in predicting the low-coverage enthalpy of adsorption of CO2 in six
metal-organic frameworks (HKUST-1, MIL-53(Al), MOF-5, UiO-66, ZIF-8 and Zn-MOF-74).
The frameworks are the guest-free structures curated for the uMOF benchmark [1].
Enthalpies are compared against experiment for the frameworks with a reference value
(currently Zn-MOF-74), and all six frameworks are used for the unphysical insertions
metric.

Each framework is relaxed (positions and cell) with janus-core, and CO2 is relaxed
in vacuum. Widom insertion is then run with FLAMES [2] at 298 K, with 20,000
random insertions of the rigid CO2 molecule into the relaxed framework. To include
framework flexibility, a 10 ps NVT (Langevin) simulation of the empty framework is
run, and Widom insertion is repeated on five snapshots taken every 1 ps after 5 ps
of equilibration, with 4,000 insertions each.

The interaction energy of every insertion is saved, and the Henry coefficient and
enthalpy of adsorption are calculated in the analysis:

.. math::

   \Delta H = \frac{\langle \Delta E \, e^{-\beta \Delta E} \rangle}
   {\langle e^{-\beta \Delta E} \rangle} - k_B T

Insertions rejected by the FLAMES overlap check have zero weight. Insertions with
an interaction energy below -1.25 eV are treated as unphysical and also given zero
weight, as they come from models predicting strongly attractive energies when CO2
overlaps with framework atoms.

Metrics
-------

1. Rigid MAE

Mean absolute error of the enthalpy of adsorption for the relaxed framework. If any
framework with a reference value fails, the metric is not calculated.

2. NVT MAE

Mean absolute error of the enthalpy of adsorption averaged over the NVT snapshots.

3. Unphysical insertions

Percentage of evaluated insertions with an interaction energy below -1.25 eV,
averaged over all frameworks.

Benchmark speed
---------------

Very slow: approximately 0.15 s per insertion for Zn-MOF-74 with mace-mp-0a
(float64, with D3) on an RTX 5090 Laptop GPU, so several hours per model.

FLAMES is not available on PyPI, and must be installed with
``uv sync --group flames``.

Data availability
-----------------

Input structures:

* uMOF benchmark [1]

Reference data:

* Experimental enthalpies of adsorption: Zn-MOF-74 [3]

[1] Jaffrelot Inizan T, Kamath P D, Elena A M, Persson K A
uMOF: A Universal Database, Benchmark, and Machine Learning Interatomic Potentials
for Metal-Organic Frameworks, arXiv:2608.28100 (2026)

[2] Oliveira F L, Saßnick H-D, Maurin G
FLAMES - A flexible and extensible code for Monte Carlo simulations of nanoporous
materials (2026), https://github.com/lipelopesoliveira/flames

[3] Queen W L, et al.
Comprehensive study of carbon dioxide adsorption in the metal-organic frameworks
M2(dobdc) (M = Mg, Mn, Fe, Co, Ni, Cu, Zn), Chem. Sci. 5, 4569 (2014),
https://doi.org/10.1039/c4sc02064b
