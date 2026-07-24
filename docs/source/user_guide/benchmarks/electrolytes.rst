============
Electrolytes
============

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
   (d_OH < 1.25 A). Reference value: 2.4, anchored to theory at 21 m
   (r2SCAN AIMD, Li128, 20 ps: 2.23; MACE potentials trained on the same
   r2SCAN dataset, converged over >2 ns: 2.35-2.39). The experimental
   partitioning of Watanabe et al., *J. Phys. Chem. B* 125, 7477 (2021)
   (1.93 water / 2.28 TFSI, neutron diffraction with isotopic substitution)
   is measured at a different composition (~18.5 m) and disagrees
   systematically with all MD-based partitionings (classical, AIMD, and
   MLIP), suggesting a method-related bias; it is therefore not used for
   the per-species references. Weight: 0.25; the "good" threshold (0.15)
   equals the uncertainty of the reference (AIMD vs converged-MLIP spread).

3. Li-O_TFSI coordination number error

   Same definition as above, restricted to TFSI oxygens (those bonded to a
   sulfur, d_OS < 1.75 A). Reference value: 1.9, anchored to theory at 21 m
   (r2SCAN AIMD: 2.02; converged r2SCAN-trained MACE: 1.89-1.91).
   Weight: 0.25; "good" threshold 0.05 (FT/TfS agreement).

4. Li-O_total coordination number error

   Absolute error in the total Li-O coordination number (water + TFSI).
   Reference value: 4.3, from theory at 21 m (r2SCAN AIMD: 4.31; converged
   r2SCAN-trained MACE: 4.27-4.29), and consistent with the experimental
   total of 4.21 +/- 0.03 at ~18.5 m (Watanabe et al. 2021), which is
   robust to the concentration difference. Weight: 0.5; "good" threshold
   0.10 (theory-experiment spread). The three CN metrics together carry
   the same total weight (1.0) as the density and S(q) families.

5. S(q) R-factor

   R-factor between the computed and experimental X-ray structure factor,
   ``sum(|S_exp - S_calc|) / sum(|S_exp|)``. S(q) is computed via dynasor
   in the Faber-Ziman convention with Cromer-Mann 4-Gaussian form factors
   (including hydrogen) and Savitzky-Golay smoothing
   (window = 5, order = 3, dq = 0.02 A^-1, i.e. a physical width of
   0.10 A^-1). Reference: SAXS data of
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

The reference data were generated with LAMMPS + symmetrix on Adastra (MI250X)
using the protocol below, at dt = 0.5 fs, T = 298.15 K, P = 1.01325 bar,
Nose-Hoover thermostat tau = 50 fs and barostat tau = 500 fs:

* p64_w170: Min -> NVT 50 ps equilibration -> NVT 50 ps production, held at the
  experimental volume throughout (no NPT), giving the trajectory used for S(q)
  and the RDF.
* p16_w42: Min -> NVT 50 ps -> NPT 200 ps, with the density averaged over the
  last 150 ps.

The janus-core recast of the same protocol is ``test_reference_md`` in
``ml_peg/calcs/electrolytes/litfsi_h2o_21m/``, marked ``very_slow``. It
regenerates both data products for any registered model, so the benchmark does
not depend on the pre-computed trajectories.

Reference data:

* Density: Gilbert et al., *J. Chem. Eng. Data* 62, 2056 (2017),
  DOI: 10.1021/acs.jced.7b00135.
* Li-O coordination numbers: theoretical references at 21 m (r2SCAN AIMD and
  r2SCAN-trained MACE potentials, Brugnoli et al., arXiv:2603.22099);
  experimental total CN at ~18.5 m: Watanabe et al., *J. Phys. Chem. B* 125,
  7477 (2021), DOI: 10.1021/acs.jpcb.1c04693.
* X-ray S(q): Zhang et al., *J. Phys. Chem. B* 125, 4501 (2021),
  DOI: 10.1021/acs.jpcb.1c02189.

Further details on the simulation protocol and MLIP assessment for this
system: Brugnoli et al., arXiv:2603.22099 (2026).
