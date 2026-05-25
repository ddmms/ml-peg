================
Alloy Metallurgy
================

The alloy metallurgy category collects benchmarks for multi-property metallic alloy workflows, starting with the Al-Cu-Mg-Zn regression suite migrated from evalpot.

Al-Zn-Cu-Mg Regression
======================

The first implementation slice covers bulk OQMD structures for the Al-Cu-Mg-Zn regression workflow. It stages the pure Al, Cu, Mg, and Zn reference structures plus selected Al-Cu and Al-Cu-Mg precipitates, computes model energies, derives elemental references, and reports formation-energy, volume-per-atom, lattice, and optional elastic errors against the DFT reference data used by the legacy evalpot plotting workflow. An opt-in solute-solute binding slice is also available for Al-matrix and Cu-matrix neighbor-shell interactions.

Data Provenance
---------------

The staged structures come from the Open Quantum Materials Database (OQMD), https://oqmd.org/, under the Creative Commons Attribution 4.0 International license. Cite:

* Saal, Kirklin, Aykol, Meredig, and Wolverton, "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)", JOM 65, 1501-1509 (2013). doi:10.1007/s11837-013-0755-4
* Kirklin, Saal, Meredig, Thompson, Doak, Aykol, Ruhl, and Wolverton, "The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies", npj Computational Materials 1, 15010 (2015). doi:10.1038/npjcompumats.2015.10

Current Scope
-------------

The initial slice is intentionally small: it validates the ML-PEG calculation, analysis, and app data flow before porting the more expensive surface, stacking-fault, GSF, antisite, cluster, triplet, and interface-energy workflows. Solute-solute bindings are present as a ``very_slow`` opt-in calculation because the legacy comparison relaxes many large FCC supercells.