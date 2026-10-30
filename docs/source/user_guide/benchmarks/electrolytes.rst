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

2. Stable trajectories

Percentage of systems completed without exploding.

During each MD run the configuration is checked periodically for non-finite
positions, energies or forces, which terminates the trajectory. A system counts
as stable only if no such failure occurs and the full 1 ns of MD is completed.
The reported value is the percentage of the 49 systems that are stable, so 100%
indicates that every trajectory ran to completion.

Benchmark speed
------------------

Very slow: tests are likely to take hours to days to run on GPU.


Data availability
-----------------

Input structures:

* 49 solid-state electrolyte structures comprising Li, Cs, Cu, and Na-containing
  ionic conductors at temperatures between 300 K and 1300 K.

Reference data:

* AIMD reference RDFs computed with VASP using the PBE functional from

* López, C., Rurali, R. & Cazorla, C. How Concerted Are Ionic Hops in
  Inorganic Solid-State Electrolytes? J. Am. Chem. Soc. 146, 8269–8279 (2024).


LIB Electrolyte Inter-Intra Properties
======================================

Summary
-------

These tests examine the model's ability to accurately predict inter- and intra- molecular properties of a common LIB battery electrolyte and solvent.
Inter-molecular forces heavily influence the prediction of density and diffusivity for liquids. Evaluate models on a mix of 200 LIB full electrolyte
and neat solvent configs across a range of densities. The following predicted properties will be tested against PBE DFT:

Intra-forces
Inter-forces
Inter-energy
Intra-virial
Inter-virial


Metrics
-------

1. RMSE (PBE)

Root mean square errors for each predicted property compared to PBE data.

All properties listed above are calculated for each structure. The intra decomposition for a frame is achieved by isolating each molecule and evaluating it with PBE.
The inter properties are then calculated by subtracting the intra property from the total property prediction. The D3 correction is applied both on the models and the PBE functional.

Computational cost
------------------

Small: tests are likely to take seconds to 10 minutes of GPU time per model.


Data availability
-----------------

Input structures:

* Built from LIB ful electrolyte (LiPF6 EC:EMC) and neat solvent (EC:EMC) configs.

Reference data:

* DFT data

  * PBE-D3(BJ)


LIB Electrolyte Volume Scans
============================

Summary
-------

Evaluate model energy predictions across battery solvent and battery electrolyte Volume Scans.

Metrics
-------

(1) Energy RMSE

Root mean square error (RMSE) between predicted and reference energy values for each volume scan config.

Volume scans consist of an initial config, where the molecules are frozen and the volume isotropically expanded or contracted.
The resulting set of configurations represent a scan across different electrolyte densities with all intra properties remaining unchanged.
The relative energy difference between densities is fully dependent on inter-molecular interactions, which heavily influence the density and diffusivity of an electrolyte.
The D3 correction is applied both on the models and the PBE functional.

Computational cost
------------------

Small: tests are likely to take seconds to 10 minutes of GPU time per model.

Data availability
-----------------

Input structures:

* Constructed using the aseMolec package https://github.com/imagdau/aseMolec.git
* PBE
