===============
Iron Properties
===============

Summary
-------

This benchmark evaluates MLIP performance on a comprehensive set of BCC iron
properties relevant to plasticity and fracture. The benchmark is based on
`Zhang et al. (2023) <https://arxiv.org/abs/2307.10072>`_, which assessed the
efficiency, accuracy, and transferability of machine learning potentials for
dislocations and cracks in iron.

Seven groups of properties are computed and compared against DFT (PBE) reference
values: equation of state, elastic constants, the Bain path, vacancy formation
energy, surface energies, generalised stacking fault energies, and
traction-separation curves.

Metrics
-------

EOS properties
^^^^^^^^^^^^^^

1. Lattice parameter error (%)

The equilibrium BCC lattice parameter :math:`a_0` is obtained by fitting a
third-order Birch-Murnaghan equation of state to 30 energy-volume points
sampled around 2.834 Å. The percentage error relative to the DFT reference
value (2.831 Å) is reported.

2. Bulk modulus error (%)

The bulk modulus :math:`B_0` is extracted from the same EOS fit. The
percentage error relative to the DFT reference value (178.0 GPa) is reported.


Elastic constants
^^^^^^^^^^^^^^^^^

3. :math:`C_{11}` error (%)

The elastic constant :math:`C_{11}` is computed using a stress-strain approach
on a 4x4x4 BCC supercell. Small positive and negative strains
(:math:`\pm 10^{-5}`) are applied along each Voigt direction, and the elastic
constants are extracted from the resulting stress differences. The percentage
error relative to the DFT reference value (296.7 GPa) is reported.

4. :math:`C_{12}` error (%)

Same as (3), for the elastic constant :math:`C_{12}`. Reference: 151.4 GPa.

5. :math:`C_{44}` error (%)

Same as (3), for the elastic constant :math:`C_{44}`. Reference: 104.7 GPa.


Vacancy formation energy
^^^^^^^^^^^^^^^^^^^^^^^^^

6. :math:`E_{\mathrm{vac}}` error (%)

A single vacancy is created in a 4x4x4 BCC supercell by removing one atom.
Atomic positions are relaxed at fixed cell volume. The vacancy formation energy
is calculated as
:math:`E_{\mathrm{vac}} = E_{\mathrm{defect}} - E_{\mathrm{perfect}} + E_{\mathrm{coh}}`,
where :math:`E_{\mathrm{coh}}` is the cohesive energy per atom. The percentage
error relative to the DFT reference value (2.02 eV) is reported.


Bain path
^^^^^^^^^

7. BCC-FCC energy difference error (meV)

The Bain path maps the continuous tetragonal distortion from BCC
(:math:`c/a = 1`) to FCC (:math:`c/a = \sqrt{2}`). For each of 65 target
:math:`c/a` ratios between 0.72 and 2.0, a tetragonally distorted cell is
created and its volume is relaxed isotropically (uniform scaling preserving the
:math:`c/a` ratio). The absolute error in the BCC-FCC energy difference
relative to the DFT reference (83.5 meV/atom) is reported.


Surface energies
^^^^^^^^^^^^^^^^

8. Surface energy MAE (J/m²)

Surface energies are computed for the (100), (110), (111), and (112) cleavage
planes. For each surface, a slab is created with vacuum and the surface energy
is calculated as
:math:`\gamma = (E_{\mathrm{slab}} - E_{\mathrm{bulk}}) / 2A`.
Atomic positions are relaxed at fixed cell shape. The mean absolute error
across all four surfaces, relative to DFT reference values
(:math:`\gamma_{100}` = 2.41, :math:`\gamma_{110}` = 2.37,
:math:`\gamma_{111}` = 2.58, :math:`\gamma_{112}` = 2.48 J/m²), is reported.


Stacking fault energies
^^^^^^^^^^^^^^^^^^^^^^^

9. Max SFE :math:`\{110\}\langle111\rangle` error (%)

The generalised stacking fault energy (GSFE) curve for the
:math:`\{110\}\langle111\rangle` slip system is computed by incrementally
displacing the upper half of a crystallographically oriented supercell along
the slip direction. The displacement covers one full Burgers vector
(:math:`b = a\sqrt{3}/2`) in 16 steps. Atoms are constrained to relax only
perpendicular to the fault plane. The percentage error in the maximum
(unstable) SFE relative to the DFT reference (0.75 J/m²) is reported.

10. Max SFE :math:`\{112\}\langle111\rangle` error (%)

Same as (9), for the :math:`\{112\}\langle111\rangle` slip system.
Reference: 1.12 J/m².


Traction-separation curves
^^^^^^^^^^^^^^^^^^^^^^^^^^

11. Max traction (100) error (%)

A traction-separation curve is computed for the (100) cleavage plane by
incrementally separating crystal halves in 0.05 Å steps up to 5.0 Å, without
atomic relaxation, and measuring forces at each step. The traction (tensile
stress) is obtained from the sum of z-forces on the upper region divided by
the cross-sectional area. The percentage error in the maximum traction relative
to the DFT reference (35.0 GPa) is reported.

12. Max traction (110) error (%)

Same as (11), for the (110) cleavage plane. Reference: 30.0 GPa.


Computational cost
------------------

Medium: tests are likely to take minutes to run on GPU, or hours on CPU for each model.
The benchmark is marked as slow and excluded from default test runs.


Data availability
-----------------

Input structures:

* All structures are generated programmatically using ASE. BCC iron unit cells and
  supercells are constructed from the equilibrium lattice parameter obtained via EOS
  fitting.

Reference data:

* DFT (PBE) reference values from:

  * Zhang, L., Csányi, G., van der Giessen, E., & Maresca, F. (2023).
    "Efficiency, Accuracy, and Transferability of Machine Learning Potentials:
    Application to Dislocations and Cracks in Iron."
    `arXiv:2307.10072 <https://arxiv.org/abs/2307.10072>`_
