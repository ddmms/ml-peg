============
Electrolytes
============

SSE-MD
======

Summary
-------

Performance in predicting the structural dynamics of 49 solid-state electrolyte (SSE)
systems via long-timescale molecular dynamics. Systems include Li, Cs, Cu, and
Na-containing ionic conductors spanning a temperature range of 300–1300 K. Radial
distribution functions (RDFs) computed from model trajectories are compared against
*ab initio* molecular dynamics (AIMD) reference data.


Metrics
-------

1. RDF Score

Minimum RDF similarity score across all systems and element pairs.

For each system, a 1 ns NVT molecular dynamics trajectory is generated using
a Nosé-Hoover chain thermostat at the target temperature. After discarding an
equilibration period of 5 ps, pairwise radial distribution functions (RDFs) are
computed for all unique element pair combinations. Each RDF is compared to the
corresponding AIMD reference using the normalised mean absolute error metric
described in Schran et al. (2021):

.. math::

   \epsilon = \frac{\sum |g_\mathrm{model}(r) - g_\mathrm{ref}(r)|}{\sum g_\mathrm{model}(r) + \sum g_\mathrm{ref}(r)}

The per-pair score is defined as :math:`1 - \epsilon`, and the per-system score
is the minimum score across all element pairs. The reported RDF Score is the
minimum across all systems. A score of 1.0 indicates perfect agreement with the
reference data.

* C. Schran, F. L. Thiemann, P. Rowe, E. A. Müller, O. Marsalek, A. Michaelides,
  "Machine learning potentials for complex aqueous systems made simple",
  Proceedings of the National Academy of Sciences 118, e2110077118 (2021).

Computational cost
------------------

High: tests are likely to take hours to days to run on GPU.


Data availability
-----------------

Input structures:

* 49 solid-state electrolyte structures comprising Li, Cs, Cu, and Na-containing
  ionic conductors at temperatures between 300 K and 1300 K.

Reference data:

* AIMD reference RDFs computed with VASP using the PBE functional from

* López, C., Rurali, R. & Cazorla, C. How Concerted Are Ionic Hops in 
Inorganic Solid-State Electrolytes? J. Am. Chem. Soc. 146, 8269–8279 (2024).

