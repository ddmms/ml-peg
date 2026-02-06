=============
Battery Electrolyte
=============

These tests are related to the model's ability to accurately predict density and diffusivity for battery electrolytes.

Inter-Intra Properties
=================

Summary
-------

Evaluate models on a mix of 200 LIB full electrolyte and neat solvent configs across a range of densities. The following predicted properties will be tested against PBE DFT:

Intra-forces
Inter-forces
Inter-energy
Intra-virial
Inter-virial

These tests are related to the model's ability to accurately predict density and diffusivity for these systems.

Metrics
-------

1. RMSE (PBE)

Root mean square errors for each predicted property compared to PBE data.

All properties listed above are calculated for each structure. The intra decomposition for a frame is achieved by isolating each molecule and evaluating it with PBE.
The intra properties are then calculated by subtracting the intra property from the total property prediction. The D3 correction is applied both on the models and the PBE functional.

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


Volume Scans
==========

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
