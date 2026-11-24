============
Electrolytes
============

LiTFSI/H2O 21 m
===============

Summary
-------

Performance in predicting structural and thermodynamic properties of the 21 molal
LiTFSI/H2O "water-in-salt" electrolyte (WiSE), in which only 2.6 water molecules
are available per ion pair: Li+ is coordinated by both water and TFSI- oxygens,
and the anions form a percolating network. The benchmark compares three
observables for the same chemistry: the NPT equilibrium density, the Li-O
coordination numbers from the radial distribution functions, and the X-ray static
structure factor S(q).

Two cubic cells are used: a small one (16 LiTFSI + 42 H2O, 382 atoms) for the NPT
density, and a large one (64 LiTFSI + 170 H2O, 1534 atoms, L = 27.4938 A at the
experimental density) for the NVT trajectories that feed the RDFs and S(q).

Protocol
--------

Both cells were generated and equilibrated once, at the experimental density,
with the procedure described in Brugnoli et al., *J. Chem. Theory Comput.* 22,
9927 (2026), SI Section 1. Every model starts from the same two configurations,
which ``test_reference_md`` downloads from the ml-peg data bucket.

MD is run with janus-core at dt = 0.5 fs, T = 298.15 K and P = 1 atm, with a
Nose-Hoover chain thermostat (tau = 50 fs) and, for NPT, the
Martyna-Tobias-Klein barostat (tau = 500 fs):

* p64_w170: NVT 50 ps equilibration -> NVT 50 ps production at the experimental
  volume (no NPT). This trajectory gives the RDFs and S(q), so that the liquid
  structure is compared at the reference density rather than at each model's
  own.
* p16_w42: NVT 50 ps equilibration -> NPT 200 ps production, with the density
  averaged over the last 150 ps. The last frame of this run is shown in the
  app when a model's bar in the density chart is clicked: it is the cell at
  the density the model settles at.

Models not trained on dispersion-corrected data are run with a D3(BJ)
correction added at runtime, as in the other ml-peg MD benchmarks, and carry the
``-D3`` suffix in the table. Models whose training data already include
dispersion (e.g. OMol25, SPICE) are run as they are.

Metrics
-------

1. Density error (weight 1.0)

   Absolute error in the NPT density vs the experimental value (1.7126 g/cm3,
   Gilbert et al., *J. Chem. Eng. Data* 62, 2056 (2017)). "Good" threshold:
   0.034 g/cm3 (2%), the RMS fluctuation of the instantaneous density of the
   382-atom cell; "bad": 0.257 g/cm3 (15%), as in the aluminosilicate glass
   density benchmark. The percent error is also reported, with the same
   thresholds, but not scored.

2. Li-O_water coordination number error (weight 0.25)

   Absolute error in the Li-O_water coordination number, computed by
   integrating the Li-O_water radial distribution function (bin 0.02 A) up to
   the first minimum of the total Li-O g(r) (R_cut = 2.83 A, from r2SCAN AIMD).
   Water oxygens are identified from the first frame as those bonded to a
   hydrogen (d_OH < 1.25 A). Reference value: 2.3. The r2SCAN AIMD at 21 m
   gives 2.347 over 0-20 ps (Li128 cell, computed with the definitions
   above). Two MACE models trained on r2SCAN calculations of this electrolyte
   (Brugnoli et al., *J. Chem. Theory Comput.* 22, 9927 (2026)) were used to
   check that the split holds over nanoseconds, with 2.296 over 17 ns for the
   model fine-tuned from mace-matpes-r2scan and 2.341 over 2.7 ns for the
   model trained from scratch, which shares no weights with the benchmarked
   models. The three estimates of each coordination number differ by
   0.05-0.06, and for this reason the references are given to one decimal;
   they are within 0.05 of the AIMD and within 0.07 of both MACE runs. The
   experimental partitioning of Watanabe et al., *J. Phys. Chem. B* 125, 7477
   (2021) (1.93 water / 2.28 TFSI, neutron diffraction with isotopic
   substitution) was measured at a different composition (~18.5 m) and
   disagrees systematically with all MD-based partitionings (classical, AIMD,
   and MLIP); it is not used for the per-species references.

   All models start from the same pre-equilibrated cell, and within the
   100 ps of NVT (50 ps equilibration and 50 ps production) the water/TFSI
   split partly retains the memory of the starting configuration. Seven
   models were also run from a packmol cell: the water and TFSI coordination
   numbers moved in opposite directions, by 0.25 and 0.20 on average
   (0.01-0.42 depending on the model), while the total changed by 0.05 on
   average (0.12 for the worst model) and the S(q) R-factor by less than
   0.02. Converging the split would require nanoseconds of MD per model.
   The "good" threshold (0.20) is about the average shift between the two
   starting cells; the two partial coordination numbers together weigh 0.5
   out of 2 in the score. The "bad" threshold (0.5) is half an oxygen.
   With a larger error, the predicted first shell is closer to a shell with
   one oxygen more (or one fewer) than to the reference.

3. Li-O_TFSI coordination number error (weight 0.25)

   Same definition, restricted to TFSI oxygens (those bonded to a sulfur,
   d_OS < 1.75 A). Reference value: 2.0, with 1.985 from r2SCAN AIMD
   (0-20 ps), and 1.977 and 1.938 from the fine-tuned and the from-scratch
   MACE models, respectively. Thresholds as for water ("good" 0.20, "bad"
   0.5).

4. Li-O_total coordination number error (weight 0)

   Absolute error in the total Li-O coordination number (water + TFSI).
   Reference value: 4.3 (4.332 in r2SCAN AIMD over 0-20 ps, 4.273 and 4.279
   for the two r2SCAN-trained MACE models), close to the experimental total
   of 4.21 +/- 0.03 at ~18.5 m (Watanabe et al. 2021). Reported and
   colour-coded ("good" 0.10, about the gap between the reference and
   experiment; "bad" 0.5, half an oxygen) but not scored, since scoring the
   sum of the two partial coordination numbers would reward models whose
   water and TFSI errors cancel.

5. S(q) R-factor (weight 0.5)

   ``R = sum(|S_exp - S_calc|) / sum(|S_exp|)`` on the experimental q grid
   from 0.51 to 11.9 A^-1. S(q) is computed with dynasor in the Faber-Ziman
   convention with Cromer-Mann 4-Gaussian form factors (including hydrogen),
   spherically averaged on a 0.02 A^-1 grid up to 13 A^-1, and smoothed with a
   Savitzky-Golay filter (window 27 points = 0.54 A^-1, order 3). The window
   is 2.4x the 2*pi/L spacing of the q grid of the periodic cell, below which
   the curve shows the quantisation of the box rather than the liquid, and it
   is the resolution at which TRAVIS returns S(q). "Good": 0.02, the
   finite-size floor (moving from 382 to 1534 atoms at fixed density shifts R
   by about 0.01-0.02); "bad": 0.21, the R-factor of a structureless fluid
   (S(q) = 1 on the same grid). The position of the first peak is computed
   but not scored. Reference: SAXS data of Zhang et al., *J. Phys. Chem. B*
   125, 4501 (2021).

Score
-----

The score is the weighted mean of the normalised metrics (linear between the
"good" and "bad" thresholds, clipped to [0, 1]) with nominal weights 2:1:1 for
density, S(q) and coordination (water 0.25 + TFSI 0.25). The density has the
largest weight because its reference is an experimental value at 21 m, while
the R-factor is compressed by the finite-size floor and the coordination
references are theoretical. For the 23 models in the first version of this
benchmark, varying the two coordination weights independently between half
and 1.5 times their nominal value changes the ranking little (Spearman
correlation with the default ranking of 0.93 or above, same first model).
Without the coordination numbers the correlation is 0.92 and the first
model does not change. With the density as the only metric the correlation
is 0.87, and four models, all within the 2% "good" threshold, share the top
score.

A model gets a score only if all its metrics are available. If the MD fails on
one cell, the calculation records it in ``failed_<cell>.json`` and still runs
the other cell; the metrics of the failed cell, and with them the score, are
NaN. A model is never scored on the metrics that survived.

Benchmark speed
---------------

Multi-day: ``test_reference_md`` runs 350 ps of MD per model (NVT 100 ps on
1534 atoms, NVT 50 ps + NPT 200 ps on 382 atoms). Measured on one A100
(80 GB) in float32: 5-190 GPU-hours per model without dispersion (median
31 h; mace-mp-0a 24 h, pet-oam-xl 187 h). The runtime D3 term costs about as
much as a medium-sized MLIP, so it roughly triples the time of the models that
need it (mace-mp-0a: 75 h).

Fast: extracting the density, RDFs and S(q) from existing trajectories takes
seconds per model.

Data availability
-----------------

Input data (``inputs/electrolytes/litfsi_h2o_21m/litfsi_h2o_21m.zip`` in the
ml-peg data bucket):

* ``p64_w170_equilibrated.xyz``: cubic cell (1534 atoms, 27.4938 A) for the NVT
  trajectories.
* ``p16_w42_equilibrated.xyz``: cell (382 atoms) for the NPT density.
* ``saxs_maginn.txt``: experimental X-ray S(q) used by the analysis.

``test_reference_md`` in ``ml_peg/calcs/electrolytes/litfsi_h2o_21m/``, marked
``multi_day``, regenerates the data for any registered model, so the benchmark
does not depend on pre-computed trajectories. The analysis reads only the small
per-model outputs of the calculation (``density.json``, ``rdf.*``,
``xray_sq.*`` and the final NPT frame ``npt_final.xyz``: a few MB for all
models). The trajectories behind the published results (about 11 GB) are
available from the authors.

Reference data:

* Density: Gilbert et al., *J. Chem. Eng. Data* 62, 2056 (2017),
  DOI: 10.1021/acs.jced.7b00135.
* Li-O coordination numbers: r2SCAN AIMD at 21 m, checked over nanoseconds
  with two r2SCAN-trained MACE potentials (Brugnoli et al.,
  *J. Chem. Theory Comput.* 22, 9927 (2026)); experimental total CN at
  ~18.5 m: Watanabe et al., *J. Phys. Chem. B* 125, 7477 (2021),
  DOI: 10.1021/acs.jpcb.1c04693.
* X-ray S(q): Zhang et al., *J. Phys. Chem. B* 125, 4501 (2021),
  DOI: 10.1021/acs.jpcb.1c02189.

Further details on the simulation protocol and the MLIP assessment for this
system: Brugnoli et al., *J. Chem. Theory Comput.* 22, 9927 (2026),
DOI: 10.1021/acs.jctc.6c00579.


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

* Ilyes Batatia et al. (2025). A foundation model for atomistic materials
  chemistry. https://doi.org/10.1063/5.0297006
* Ioan-Bogdan Magdău et al. (2023). Machine learning force fields for
  molecular liquids: Ethylene Carbonate/Ethyl Methyl Carbonate binary
  solvent. https://doi.org/10.1038/s41524-023-01100-w

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

* Ilyes Batatia et al. (2025). A foundation model for atomistic materials
  chemistry. https://doi.org/10.1063/5.0297006
* Ioan-Bogdan Magdău et al. (2023). Machine learning force fields for
  molecular liquids: Ethylene Carbonate/Ethyl Methyl Carbonate binary
  solvent. https://doi.org/10.1038/s41524-023-01100-w

Computational cost
------------------

Small: tests are likely to take seconds to 10 minutes of GPU time per model.

Data availability
-----------------

Input structures:

* Constructed using the aseMolec package https://github.com/imagdau/aseMolec.git
* PBE
