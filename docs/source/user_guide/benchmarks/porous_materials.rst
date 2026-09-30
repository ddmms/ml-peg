================
Porous Materials
================

MOF heat capacity
=================

Summary
-------

Performance in predicting the isochoric heat capacity of metal-organic
frameworks (MOFs) at 300 K from the phonon density of states, compared against
experimental calorimetry.

This benchmark and :ref:`mof-ins-spectra` are scored from a single shared
phonon calculation, ``ml_peg/calcs/porous_materials/mof_phonons``. Running that
calculation once produces the outputs for both.

Metrics
-------

1. Cv MAE

Mean absolute error of the isochoric heat capacity at 300 K, in J/g/K.

Each framework is relaxed with the LBFGS optimiser, with the cell relaxed
through a ``FrechetCellFilter``, to a maximum force component of 1e-8 eV/Å or
1000 optimisation steps, whichever is reached first. The tolerance is
deliberately tight: residual forces in the starting structure appear as
spurious imaginary modes in the finite-difference phonons and would otherwise
contaminate the imaginary-mode metric.

No symmetry constraint is applied, so each framework is free to relax away
from its input space group. This makes the imaginary-mode metric a measure of
unconstrained dynamical stability, but it also means phonopy determines the
space group from the relaxed structure rather than the input, so the
supercell and displacement set may differ between models for the same
framework.

Force constants are then obtained by finite displacements of 0.01 Å in a
diagonal supercell chosen so that every supercell lattice vector is at least
20 Å, and symmetrised. The existing ``bulk_crystal/phonons`` helpers
(``init_phonopy_from_ref``, ``get_fc2_and_freqs``) are reused rather than
duplicated. Thermal properties are evaluated on an 11×11×11 q-mesh. phonopy
reports the heat capacity per mole of primitive cells, so it is divided by the
primitive cell mass to give J/g/K, which makes the value independent of the
choice of cell.

2. Imaginary modes MAE

Mean q-point-weighted percentage of imaginary phonon modes, over every
framework in the set.

The percentage is taken directly from phonopy's thermal-property
bookkeeping::

    100 * (number_of_modes - number_of_integrated_modes) / number_of_modes

Both counts are q-point-weighted, and phonopy integrates only modes with a
frequency strictly greater than its cutoff, which is left at zero. The
excluded modes are therefore exactly the imaginary ones, and the heat
capacity and this metric describe the same set of modes.

When Γ lies on the mesh, its three acoustic branches are numerically zero and
may land on either side of the strict cutoff. With symmetrised force constants
they are typically very slightly positive and are integrated, but if they come
out non-positive they are excluded and add a small floor of three q-point
weights out of the total.

Every framework in the set has been synthesised, so it is dynamically stable
and the expected percentage is zero. There is no per-structure reference
value, so the metric is the mean deviation from that expected zero. It is
reported identically in both MOF phonon benchmarks.

Computational cost
------------------

High: supercells range from roughly 450 to 2200 atoms, and each framework
requires a full finite-displacement set. A GPU is strongly recommended.

Data availability
-----------------

Input structures:

* 15 MOF CIFs, downloaded from the ML-PEG data bucket as
  ``inputs/porous_materials/mof_phonons/mof_phonons.zip``. Set the
  ``ML_PEG_MOF_PHONONS_DATA`` environment variable to an unpacked copy to
  work offline.

Reference data:

* Experimental heat capacities at 300 K, extracted from the primary
  literature, in ``heat_capacity_reference.json`` within that archive. All
  seven reference values correspond to a benchmark structure and are scored.
  The calculation step copies the reference files to ``outputs/reference/``,
  so the analysis step does not need to reach the data bucket.

.. _mof-ins-spectra:

MOF INS spectra
===============

Summary
-------

Performance in predicting the inelastic neutron scattering (INS) spectra of
MOFs, compared against digitised experimental spectra.

Metrics
-------

1. Wasserstein distance

Wasserstein-1 distance between the simulated and measured spectra, in meV,
averaged over frameworks.

INS intensities are in arbitrary units and the reference spectra are digitised
from published figures on their own irregular energy grids, so neither the
intensity scale nor the sampling can be compared directly. The comparison
therefore proceeds in four steps:

1. Clip both spectra at zero, sort them by energy, and collapse any repeated
   energies, since digitised traces contain both negative excursions and
   duplicated abscissae.
2. Restrict both to the comparison window the reference declares in
   ``window_meV``. Every reference shipped here declares one; the overlap of
   the two energy ranges is only a fallback.
3. Resample both onto a single uniform grid spanning that window, at the finer
   of the two native resolutions.
4. Normalise each by its own area over that grid, so both become distributions
   over energy.

Step 4 is what puts the measurement and the calculation on a common intensity
footing, and is equivalent to scaling the measured trace until its area
matches the calculated one. A per-spectrum scale factor applied beforehand is
absorbed by it and cannot change the score.

Step 2 is the step that decides what the metric measures, and the windows are
deliberately narrow: 25-60 meV for every framework except ZIF-4 (80-200) and
MIL-53 (40-90). The reason is physical rather than presentational. The
Euphonic backend returns an incoherent-approximation weighted density of
states, which keeps full intensity in the C-H and O-H stretch bands near
370-470 meV. A measured TOSCA spectrum suppresses that region through the
Debye-Waller factor and the instrument's kinematic trajectory, so the
calculation carries spectral weight there that the measurement structurally
cannot. Comparing over the full overlap makes the distance report that
missing suppression, which is a property of the backend, not of the model
under test. On MIL-53, NOTT-300 and ZIF-8 the full-overlap distances were
6.6, 17.3 and 38.5 meV against 3.8, 1.6 and 3.1 meV over these windows, and
the computed spectra already reproduce the measured peak positions in the
fingerprint region to within about 2 meV.

Adding Debye-Waller suppression and the instrument trajectory to the backend
would remove the need for the windows and is the natural next step; the Abins
backend models both, so selecting it where Mantid is available is the
physically faithful route.

Step 3 is not cosmetic. Digitised grids can be an order of magnitude denser in
some regions than others, and weighting raw intensities on the native grid
would count densely sampled regions more heavily than sparsely sampled ones.

The Wasserstein-1 distance between the resulting distributions measures how far
spectral weight has to move, in meV, to turn one into the other, and so
penalises systematically over- or under-stiff modes rather than differences in
overall intensity.

2. Imaginary modes MAE

As defined for the MOF heat capacity benchmark above, and computed from the
same phonon calculation.

INS backends
------------

The simulated spectrum is produced by one of two interchangeable backends:

* **Euphonic** (default). The incoherent-approximation neutron-weighted
  density of states, summed over atoms, which is the one-phonon powder INS
  intensity. Installed with the ``ins`` extra::

      pip install ml-peg[ins]

* **Abins**. ``mantid.simpleapi.Abins`` with the TOSCA instrument in
  backscattering, second-order quantum events, autoconvolution and total
  cross-section scaling. This is closer to a measured TOSCA spectrum, but
  Mantid is only distributed through conda and so cannot be a dependency of
  ML-PEG.

``INS_BACKEND`` in ``calc_mof_phonons.py`` selects between them; its default
of ``"auto"`` uses Abins when Mantid is importable and Euphonic otherwise.

.. warning::

    The two backends do not produce numerically identical spectra, so
    Wasserstein distances are only comparable between models scored with the
    same backend. The backend used is recorded in each ``<mof>_ins.json``.

Computational cost
------------------

High: shares the phonon calculation with the MOF heat capacity benchmark, so
running both costs little more than running one.

Data availability
-----------------

Input structures:

* As for the MOF heat capacity benchmark.

Reference data:

* Experimental INS spectra for seven frameworks (MIL-140A, MIL-53, MOF-5,
  NOTT-300, ZIF-4, ZIF-7 and ZIF-8), digitised from the primary literature,
  in ``ins_reference.json.gz`` within the same archive. Published DFT spectra
  are packaged alongside them for a subset of frameworks, and can be selected
  instead through ``REFERENCE_TYPE`` in ``analyse_mof_ins.py``.

.. note::

    The reference records were extracted from the literature by a combined
    automated and manual curation pass. Each retains its source DOI and figure
    caption, and known issues with individual records are listed under
    ``known_issues`` in the reference files.

MOF bulk modulus
================

Summary
-------

Performance in predicting the bulk modulus of metal-organic frameworks from
an energy-volume scan, compared against experimental values.

This benchmark has its own calculation, ``mof_bulk_modulus``; it does not
share the phonon run.

Metrics
-------

1. Bulk modulus MAE

Mean absolute error of the bulk modulus, in GPa.

Each framework is relaxed as in the phonon benchmark (LBFGS, full cell, no
symmetry constraint, to 1e-8 eV/Å or 1000 steps). Its cell is then scaled
isotropically over ±3 % in volume (0.97 V\ :sub:`0` to 1.03 V\ :sub:`0`) at
7 points, with the internal coordinates re-relaxed to 1e-6 eV/Å at each fixed
cell. A Birch-Murnaghan equation of
state is fitted to the resulting energy-volume curve with
``ase.eos.EquationOfState``, the same routine the
``bulk_crystal/energy_volume_curves_metals`` benchmark uses, and the
experimental references were themselves obtained from Birch-Murnaghan fits.
The strain range is kept modest because MOFs are compliant and larger strains
risk crossing a pressure-induced structural transition.

Computational cost
------------------

Moderate: one full relaxation plus seven fixed-cell relaxations per framework,
with no supercell. Cheaper per framework than the phonon benchmark.

Data availability
-----------------

Input structures:

* The same archive as the phonon benchmarks.

Reference data:

* Experimental bulk moduli in ``bulk_modulus_reference.json``. Seven
  frameworks are scored: HKUST-1, MIL-125, UiO-66, UiO-66-Ce, UiO-66-Hf,
  ZIF-4 and ZIF-8.

.. note::

    Every bundled CIF is a guest-free framework. Where a source reports
    several values for different guest loadings, only the evacuated value is
    used and the others are recorded under ``not_applicable_values``. ZIF-4
    is the case in point: of its five experimental values spanning
    2.0-16.5 GPa, the three measured on solvent-occupied crystals do not
    describe the bundled structure, while the two evacuated determinations
    agree at 2.01 and 2.6 GPa. Similarly, where two independent
    determinations exist for one framework, the value explicitly captioned
    for that material is used and the other is kept under
    ``alternative_values`` — UiO-66 has 37.9 GPa captioned for Zr-UiO-66 and
    41.0 GPa from the reference row of a force-field comparison table.

    **MIL-53** is the one framework retained but flagged ``excluded``, with a
    reason, so the omission is auditable rather than silent: it is a
    breathing framework and its 0.35 GPa describes the soft large-pore regime
    around a pressure-induced transition, which is not comparable to a fit
    about a single relaxed structure.

    **UiO-67 is not scored**: the reference database holds only DFT bulk
    moduli for it (10.1021/jz4002345, 17.15 GPa), not experimental ones.
    MOF-5 and MOF-177 are in the same position (10.1002/pssb.201100634).
