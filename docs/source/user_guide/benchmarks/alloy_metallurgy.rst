================
Alloy Metallurgy
================

Al-Zn-Cu-Mg Regression
=======================

Summary
-------

Performance on a multi-property Al-Cu-Mg-Zn alloy metallurgy regression suite
benchmarked against DFT (PBE) reference data from the OQMD and the evalpot
legacy dataset. The benchmark covers bulk formation energies, lattice geometry,
elastic constants, solute-solute binding curves, surface energies, stacking fault
energies, generalised stacking fault (GSF) curves, and solute-stacking fault
interaction energies, targeting precipitate phases (Θ and Θ'') relevant to
Al-alloy design.


Metrics
-------

Bulk properties
^^^^^^^^^^^^^^^

Calculated for 8 OQMD structures (Al, Al-Cu, Mg, Mg-Zn, Cu, Zn, and two
Al-Cu-Mg precipitate cells). Each structure is relaxed with the LBFGS
optimiser with FrechetCellFilter applied, until the maximum force component is
below 1×10⁻⁶ eV/Å or 1000 steps are reached. Elemental reference energies are
derived from the relaxed pure-element structures and used to compute formation
energies.

1. Formation Energy MAE (meV/atom)

   Mean absolute error of the DFT formation energy per atom.

2. Volume per Atom MAE (Å³/atom)

   Mean absolute error of the relaxed volume per atom.

3. Lattice Constant MAE (Å)

   Mean absolute error of the relaxed lattice constants.

4. Beta Angle MAE (°)

   Mean absolute error of the relaxed monoclinic beta lattice angle.


Surface and fault energies
^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculated for FCC Al (OQMD 8100) and Cu (OQMD 635950) and HCP Mg (OQMD 9226)
and Zn (OQMD 122929). Each reference structure is first relaxed fully (atoms
and cell) with fmax=1×10⁻⁶ eV/Å.

5. Surface Energy MAE (J/m²)

   Mean absolute error of surface energies γ = (E\ :sub:`slab` − N × E\ :sub:`bulk`) / (2A).
   FCC surfaces: (111), (100), (110) for Al and Cu.
   HCP surfaces: (0001), (10m10), (11m20), (1m101), (10m12), (11m21), (11m22) for
   Mg and Zn.

6. Stacking Fault Energy MAE (mJ/m²)

   Mean absolute error of the stable (2/3 Burgers vector) and unstable (5/6
   Burgers vector) stacking fault energies for FCC Al and Cu. Atoms are allowed
   to relax in the direction perpendicular to the fault plane (fmax=5×10⁻³ eV/Å,
   up to 100 steps).

7. GSF Energy MAE (mJ/m²)

   Mean absolute error of the generalised stacking fault energy curves for the
   Θ precipitate {111} slip system (12 displacement points, base cell AIIDA_339739)
   and the Θ'' precipitate {0m11} slip system (8 displacement points, base cell
   AIIDA_481617). Atoms are allowed to relax perpendicular to the fault plane;
   for Θ'' the cell z-dimension is also relaxed (fmax=5×10⁻³ eV/Å, up to 200
   steps per point).

8. Solute-Stacking Fault MAE (mJ/m²)

   Mean absolute error of the solute-stacking fault interaction energies for
   Cu, Mg, Zn, and Si solutes in an Al-FCC matrix (4×4 in-plane, 4 z-planes)
   and Al solutes in a Cu-FCC matrix (3×4 in-plane, 4 z-planes).


Elastic constants (slow)
^^^^^^^^^^^^^^^^^^^^^^^^

Calculated for all 8 bulk structures after full cell relaxation.
Elastic constants are derived from a stress-strain approach using strain
magnitude ±0.5%.

9. Bulk Modulus MAE (GPa)

   Mean absolute error of the Voigt-Reuss-Hill bulk modulus.

10. Shear Modulus MAE (GPa)

    Mean absolute error of the Voigt-Reuss-Hill shear modulus.

11. Elastic Constant MAE (GPa)

    Element-wise mean absolute error across all 21 independent C\ :sub:`ij`
    components of the 6×6 Voigt elastic tensor.


Solute-solute binding (very slow)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Binding energy curves as a function of solute-solute separation distance.
Pure-element FCC supercells are constructed and fully relaxed, then pairs of
solutes are introduced and the binding energy is computed for all symmetry-
inequivalent neighbour shells up to max_index=8.

12. Solute-Solute Binding MAE (eV)

    Mean absolute error of the solute-solute binding energies. Pairs computed:
    Zn-Zn, Zn-Cu, Zn-Mg, Zn-Vac, Cu-Cu, Cu-Mg, Cu-Vac, Mg-Mg, Mg-Vac,
    Vac-Vac in Al-FCC (4×4×4 supercell); Al-Al, Al-Vac, Vac-Vac in Cu-FCC
    (3×3×3 supercell).


Computational cost
------------------

* **Bulk properties** (``test_alzncumg_regression``): Low — 8 bulk cell relaxations;
  likely less than a minute on GPU.

* **Surface and fault energies** (``test_alzncumg_fault_surfaces``): Medium — surface
  slab and stacking fault calculations across ~30 surface/fault/GSF combinations;
  likely minutes on GPU.

* **Elastic constants** (``test_alzncumg_elasticity``, marked ``slow``): Medium —
  stress-strain elastic tensor for 8 structures; likely minutes to tens of minutes
  on GPU.

* **Solute-solute binding** (``test_alzncumg_solute_solute``, marked ``very_slow``):
  High — many FCC supercell relaxations per solute pair; likely hours on GPU.


Data availability
-----------------

Input structures:

* OQMD structures: Al (8100), Al-Cu alloy (635950), Mg (9226), Mg-Zn (122929),
  Cu (695020), Zn (10434) — from the Open Quantum Materials Database (https://oqmd.org/)
  under the Creative Commons Attribution 4.0 International licence.

* Non-OQMD precipitate cells: Θ'' (NOTINOQMD_00001) and Θ (NOTINOQMD_00002) —
  from the legacy evalpot dataset.

* Special GSF base cells: AIIDA_339739 (Θ {111}) and AIIDA_481617 (Θ'' {0m11}) —
  VASP POSCAR cells from AiiDA DFT workflows included in the S3 data bundle.

Reference data:

* DFT (PBE) values from the evalpot legacy dataset, stored in
  ``alzncumg_regression/references/DFT.json``.

* OQMD:

  * Saal, J.E., Kirklin, S., Aykol, M., Meredig, B., and Wolverton, C.,
    "Materials Design and Discovery with High-Throughput Density Functional
    Theory: The Open Quantum Materials Database (OQMD)", *JOM* 65, 1501–1509
    (2013). https://doi.org/10.1007/s11837-013-0755-4

  * Kirklin, S., Saal, J.E., Meredig, B., Thompson, A., Doak, J.W., Aykol, M.,
    Rühl, S., and Wolverton, C., "The Open Quantum Materials Database (OQMD):
    assessing the accuracy of DFT formation energies", *npj Computational
    Materials* 1, 15010 (2015). https://doi.org/10.1038/npjcompumats.2015.10
