==================
WiSE electrolytes
==================

LiTFSI/H2O 21 m
===============

Summary
-------

Performance in predicting structural and thermodynamic properties of the 21 molal
LiTFSI/H2O "water-in-salt" electrolyte (WiSE). The benchmark consolidates three
experimental observables for the same chemistry: NPT equilibrium density, Li-O
coordination from the radial distribution function, and the X-ray static
structure factor S(q).

Two simulation cells are used: a small cell (16 LiTFSI + 42 H2O, 382 atoms) for
the NPT density, and a large cubic cell (64 LiTFSI + 170 H2O, 1534 atoms,
27.4938 A) for the NVT trajectories that feed the RDF and S(q).

Metrics
-------

1. Density error

   Absolute error in NPT density vs the experimental value
   (1.7126 g/cm3, Gilbert et al., *J. Chem. Eng. Data* 62, 2056 (2017)).
   The density is the average over the production NPT trajectory at
   298.15 K and 1 atm.

2. Li-O_water coordination number error

   Absolute error in the Li-O_water coordination number, computed by
   integrating the Li-O_water radial distribution function up to the first
   minimum (R_cut = 2.83 A, from r2SCAN AIMD reference). Water oxygens are
   identified from the first frame as those bonded to a hydrogen
   (d_OH < 1.25 A). Reference value: 2.0 (Watanabe et al.,
   *J. Phys. Chem. B* 125, 7477 (2021), neutron diffraction with isotopic
   substitution at ~18.5 m).

3. Li-O_TFSI coordination number error

   Same definition as above, restricted to TFSI oxygens (those bonded to a
   sulfur, d_OS < 1.75 A). Reference value: 2.0 (same source).

4. S(q) R-factor

   R-factor between the computed and experimental X-ray structure factor,
   ``sum(|S_exp - S_calc|) / sum(|S_exp|)``. S(q) is computed via dynasor
   in the Faber-Ziman convention with Cromer-Mann 4-Gaussian form factors
   (including hydrogen) and Savitzky-Golay smoothing
   (window = 27, order = 3, dq = 0.02 A^-1). Reference: SAXS data of
   Zhang et al., *J. Phys. Chem. B* 125, 4501 (2021).

Computational cost
------------------

High: production trajectories require a GPU MD code (e.g. LAMMPS + symmetrix
or Janus + MACE) and several hours per model on a single MI250X / A100 GPU.
Re-extracting metrics from pre-computed trajectories is cheap (seconds per
model).

Data availability
-----------------

Input structures:

* p64_w170 cubic cell (1534 atoms, 27.4938 A) for NVT trajectories.
* p16_w42 cell (382 atoms) for NPT density.

Production trajectories were generated with LAMMPS + symmetrix on Adastra
(MI250X) using a MACE foundation model. The reference Janus recast of the
same protocol (Min -> NVT 50 ps -> NPT 200 ps, dt = 0.5 fs, T = 298.15 K,
P = 1.01325 bar, NH thermostat tau = 50 fs, NH barostat tau = 500 fs) lives
in ``ml_peg/calcs/wise_electrolytes/md_reference/``.

Reference data:

* Density: Gilbert et al., *J. Chem. Eng. Data* 62, 2056 (2017),
  DOI: 10.1021/acs.jced.7b00135.
* Li-O coordination numbers: Watanabe et al., *J. Phys. Chem. B* 125, 7477
  (2021), DOI: 10.1021/acs.jpcb.1c04693.
* X-ray S(q): Zhang et al., *J. Phys. Chem. B* 125, 4501 (2021),
  DOI: 10.1021/acs.jpcb.1c02189.

Further details on the simulation protocol and MLIP assessment for this
system: L. Brugnoli, arXiv:2603.22099 (2026).
