=============
Bulk Crystals
=============

Lattice constants
=================

Summary
-------

Performance in evaluating lattice constants for 23 solids, including pure elements,
binary compounds, and semiconductors.


Metrics
-------

1. MAE (Experimental)

Mean lattice constant error compared to experimental data

For each formula, a bulk crystal is built using the experimental lattice constants and
lattice type for the initial structure. This structure is optimised for each model
using the LBFGS optimiser, with the FrechetCellFilter applied to allow optimisation of
the cell, until the largest absolute Cartesian component of any interatomic force is
less than 0.03 eV/Å. The lattice constants of this optimised structure are then
compared to experimental values.


2. MAE (PBE)

Mean lattice constant error compared to PBE data

Same as (1), but optimised lattice constants are compared to reference PBE data.


Computational cost
------------------

Low: tests are likely to less than a minute to run on CPU.


Data availability
-----------------

Input structures:

* Built from experimental lattice constants from various sources

Reference data:

* Experimental data same as input data

* DFT data

  * Batatia, I., Benner, P., Chiang, Y., Elena, A.M., Kovács, D.P., Riebesell, J.,
    Advincula, X.R., Asta, M., Avaylon, M., Baldwin, W.J. and Berger, F., 2025. A
    foundation model for atomistic materials chemistry. The Journal of Chemical
    Physics, 163(18).
  * PBE-D3(BJ)


Elasticity
==========

Summary
-------

Bulk and shear moduli calculated for 12122 bulk crystals from the materials project.


Metrics
-------

(1) Bulk modulus MAE

Mean absolute error (MAE) between predicted and reference bulk modulus (B) values.

MatCalc's ElasticityCalc is used to deform the structures with normal (diagonal) strain
magnitudes of ±0.01 and ±0.005 for ϵ11, ϵ22, ϵ33, and off-diagonal strain magnitudes of
±0.06 and ±0.03 for ϵ23, ϵ13, ϵ12. The Voigt-Reuss-Hill (VRH) average is used to obtain
the bulk and shear moduli from the stress tensor. Both the initial and deformed
structures are relaxed with MatCalc's default ElasticityCalc settings. For more information, see
`MatCalc's ElasticityCalc documentation
<https://github.com/materialsvirtuallab/matcalc/blob/main/src/matcalc/_elasticity.py>`_.

Analysis excludes materials with:
    * B ≤ 0, B > 500 and G ≥ 0, G > 500 structures.
    * H2, N2, O2, F2, Cl2, He, Xe, Ne, Kr, Ar
    * Materials with density < 0.5 (less dense than Li, the lowest density solid element)

(2) Shear modulus MAE

Mean absolute error (MAE) between predicted and reference shear modulus (G) values

Calculated alongside (1), with the same exclusion criteria used in analysis.


Computational cost
------------------

High: tests are likely to take hours-days to run on GPU.


Data availability
-----------------

Input structures:

* 1. De Jong, M. et al. Charting the complete elastic properties of
  inorganic crystalline compounds. Sci Data 2, 150009 (2015).
* Dataset release: mp-pbe-elasticity-2025.3.json.gz from the Materials Project database.

Reference data:

* Same as input data
* PBE


Diamond phonons
===============

Summary
-------

Performance in predicting the phonon dispersion of bulk diamond (carbon).

The benchmark evaluates the accuracy of phonon frequencies along a fixed
high-symmetry path in the Brillouin zone for a single reference crystal
structure of diamond.


Metrics
-------

1. Band MAE

Mean absolute error (MAE) between predicted and reference phonon frequencies.

For bulk diamond, the phonon band structure is computed for each model along the same
q-point path as the reference calculation. At each q-point, the six phonon frequencies
are compared to the reference frequencies after sorting the modes to avoid branch
labelling ambiguities. The MAE is evaluated over all q-points and all phonon branches.


2. Band RMSE

Root mean squared error (RMSE) between predicted and reference phonon frequencies.

The RMSE is computed using the same sorted, mode-unlabelled comparison procedure as in
(1), over all q-points and all phonon branches.


Computational cost
------------------

Medium: tests typically take a few minutes to run on CPU.


Data availability
-----------------

Input structures (https://github.com/7radians/ml-peg-data/tree/main/diamond_data):

* A primitive bulk diamond unit cell containing two carbon atoms, used to generate a
  phonopy displacement dataset on a 4×4×4 supercell.

Reference data:

* DFT phonon band structure for bulk diamond along a fixed high-symmetry path, provided
  as ``dft_band.npz``.
* RSCAN.


Ti64 phonons
============

Summary
-------

Performance in predicting phonon dispersions, vibrational densities of states (DOS/PDOS),
and thermodynamic Helmholtz free energies for a suite of 10 Ti–Al–V alloy phases.

Each case is evaluated by comparing ML-predicted phonon frequencies to CASTEP reference
phonon frequencies along a fixed high-symmetry q-path. For a subset of cases,
Helmholtz free-energy errors per atom are additionally reported.


Metrics
-------

1. Dispersion RMSE (mean)

   Mean root mean squared error (RMSE) between predicted and reference phonon frequencies,
   averaged over 10 Ti64 cases.

   For each case, reference phonon frequencies are parsed from CASTEP ``.castep`` outputs
   along a fixed high-symmetry q-path. The structure is then relaxed for each model using
   the LBFGS optimiser (maximum 10000 steps, ``fmax=0.001``). Phonon frequencies are computed
   using finite displacements in a 2×2×2 supercell with a displacement magnitude of 0.02 Å and
   ``plusminus=True``. The reference dispersion is linearly interpolated onto an inferred ML
   path-coordinate grid spanning the same path (a uniform grid with the same number of
   q-points as the ML dispersion), and the RMSE is evaluated over all q-points and all
   phonon branches.

2. Dispersion RMSE (max)

   Maximum per-case dispersion RMSE (in THz) over the 10 Ti64 cases.

   Computed as in (1), but taking the maximum RMSE value across cases.

3. ω_avg MAE

   Mean absolute error (MAE) in the average phonon frequency ω_avg over the 10 Ti64 cases.

   For each case, ω_avg is computed as the arithmetic mean of all phonon frequencies after
   interpolating the reference dispersion onto the inferred ML grid. The per-case absolute error is
   then averaged across cases. Frequencies are averaged as stored; if imaginary modes are present
   as negative values, they contribute directly.

4. ΔF (0 K) mean

   Mean absolute error in Helmholtz free energy at 0 K, reported as eV/atom, over the
   subset of cases where thermodynamic outputs are available.

   For applicable cases, CASTEP q-point phonon frequencies and q-point weights are parsed
   from CASTEP qpoints ``.castep`` outputs. A reference Helmholtz free energy is computed
   in the harmonic approximation by combining a weighted zero-point energy contribution
   with a weighted thermal free-energy contribution evaluated on a dense temperature grid
   (2000 points spanning 0–2000 K) and interpolated to the ML temperatures. The absolute
   difference between ML and reference free energy at 0 K is divided by the number of atoms
   and averaged across thermodynamics-enabled cases. Weights are taken directly from CASTEP;
   no explicit renormalization is applied.

5. ΔF (2000 K) mean

   Mean absolute error in Helmholtz free energy at 2000 K, reported as eV/atom, over the
   subset of cases where thermodynamic outputs are available.

   Computed as in (4), but using the final temperature point (2000 K).


Computational cost
------------------

Medium: dispersion, DOS/PDOS and thermodynamic calculations typically take minutes per model on CPU.
Thermodynamic calculations are enabled for a 7/10 subset of cases.


Data availability
-----------------

Full details on the data and benchmark:

* Allen, C. S. & Bartók, A. P. Multi-phase dataset for Ti and Ti-6Al-4V.
       Preprint at https://arxiv.org/abs/2501.06116 (2025).

* Radova, M., Stark, W. G., Allen, C. S., Maurer, R.J. & Bartók, A. P.
       Fine-tuning foundation models of materials interatomic potentials
       with frozen transfer learning.
       npj Comput Mater 11, 237 (2025).
       https://doi.org/10.1038/s41524-025-01727-x

Input structures (https://github.com/7radians/ml-peg-data/tree/main/ti64_data):

* CASTEP ``.castep`` outputs providing reference phonon dispersions along fixed
  high-symmetry q-paths for 10 Ti–Al–V alloy cases.
* Corresponding CASTEP qpoints ``.castep`` outputs (subset) providing q-point phonon
  frequencies and weights for thermodynamic reference reconstruction.

Reference data:

* CASTEP phonon dispersions parsed from ``.castep`` outputs (q-path dispersion).
* CASTEP q-point phonon frequencies and weights parsed from qpoints ``.castep`` outputs
  (subset), used to compute reference Helmholtz free energies in the harmonic
  approximation.
* PBE

Computational environment
-------------------------

Ti64 phonon calculations were run as a single process on CPU on an
x86_64 machine (11th Gen Intel(R) Core(TM) i5-1145G7; 4 cores / 8 threads). No explicit
parallel execution (MPI or multiprocessing) was used in the benchmark driver.
