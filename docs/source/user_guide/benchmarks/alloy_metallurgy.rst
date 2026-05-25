================
Alloy Metallurgy
================

The alloy metallurgy category collects benchmarks for multi-property metallic alloy workflows, starting with the Al-Cu-Mg-Zn regression suite migrated from evalpot.

Al-Zn-Cu-Mg Regression
======================

The first implementation slice covers bulk OQMD structures for the Al-Cu-Mg-Zn regression workflow. It stages the pure Al, Cu, Mg, and Zn reference structures, computes model energies, derives elemental references, and reports formation-energy and volume-per-atom errors against the DFT reference data used by the legacy evalpot plotting workflow.

Data Provenance
---------------

The staged structures come from the Open Quantum Materials Database (OQMD), https://oqmd.org/, under the Creative Commons Attribution 4.0 International license. Cite:

* Saal, Kirklin, Aykol, Meredig, and Wolverton, "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)", JOM 65, 1501-1509 (2013). doi:10.1007/s11837-013-0755-4
* Kirklin, Saal, Meredig, Thompson, Doak, Aykol, Ruhl, and Wolverton, "The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies", npj Computational Materials 1, 15010 (2015). doi:10.1038/npjcompumats.2015.10

Current Scope
-------------

The initial slice is intentionally small: it validates the ML-PEG calculation, analysis, and app data flow before porting the more expensive solute, surface, stacking-fault, GSF, antisite, cluster, triplet, and interface-energy workflows.