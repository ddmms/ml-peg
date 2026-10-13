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
with the procedure described in Brugnoli et al., arXiv:2603.22099 (SI, Section
1). Every model starts from the same two configurations, which
``test_reference_md`` downloads from the ml-peg data bucket.

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
   hydrogen (d_OH < 1.25 A). Reference value: 2.4, anchored to theory at 21 m
   (r2SCAN AIMD, Li128, 20 ps: 2.23; MACE potentials trained on the same
   r2SCAN dataset, converged over >2 ns: 2.35-2.39). The experimental
   partitioning of Watanabe et al., *J. Phys. Chem. B* 125, 7477 (2021)
   (1.93 water / 2.28 TFSI, neutron diffraction with isotopic substitution)
   is measured at a different composition (~18.5 m) and disagrees
   systematically with all MD-based partitionings (classical, AIMD, and
   MLIP), so it is not used for the per-species references. "Good": 0.15,
   the uncertainty of the reference (AIMD vs converged-MLIP spread); "bad": 0.5.

3. Li-O_TFSI coordination number error (weight 0.25)

   Same definition, restricted to TFSI oxygens (those bonded to a sulfur,
   d_OS < 1.75 A). Reference value: 1.9 (r2SCAN AIMD: 2.02; converged
   r2SCAN-trained MACE: 1.89-1.91). "Good": 0.05, the agreement between the
   two r2SCAN-trained potentials; "bad": 0.5.

4. Li-O_total coordination number error (weight 0)

   Absolute error in the total Li-O coordination number (water + TFSI).
   Reference value: 4.3 (r2SCAN AIMD: 4.31; converged r2SCAN-trained MACE:
   4.27-4.29), consistent with the experimental total of 4.21 +/- 0.03 at
   ~18.5 m (Watanabe et al. 2021). Reported and colour-coded ("good" 0.10,
   the theory-experiment spread; "bad" 0.5) but not scored: it is the sum of
   the two partial coordination numbers, and would reward models that miss
   water and TFSI in opposite directions.

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
density, S(q) and coordination (water 0.25 + TFSI 0.25). The density is the
only observable that is both experimental and measured at this composition;
the R-factor is compressed by the finite-size floor, and the coordination
references are theoretical. The ranking is insensitive to the choice: the
Spearman correlation between the schemes tried, including density alone, is
above 0.97.

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
* Li-O coordination numbers: theoretical references at 21 m (r2SCAN AIMD and
  r2SCAN-trained MACE potentials, Brugnoli et al., arXiv:2603.22099);
  experimental total CN at ~18.5 m: Watanabe et al., *J. Phys. Chem. B* 125,
  7477 (2021), DOI: 10.1021/acs.jpcb.1c04693.
* X-ray S(q): Zhang et al., *J. Phys. Chem. B* 125, 4501 (2021),
  DOI: 10.1021/acs.jpcb.1c02189.

Further details on the simulation protocol and the MLIP assessment for this
system: Brugnoli et al., arXiv:2603.22099 (2026).


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
