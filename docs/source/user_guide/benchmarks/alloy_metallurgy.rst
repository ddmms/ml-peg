================
Alloy Metallurgy
================

The alloy metallurgy category collects benchmarks for multi-property metallic alloy workflows, starting with the Al-Cu-Mg-Zn regression suite migrated from evalpot.

Al-Zn-Cu-Mg Regression
======================

The first implementation slice covers bulk OQMD structures for the Al-Cu-Mg-Zn regression workflow. It stages the pure Al, Cu, Mg, and Zn reference structures plus selected Al-Cu and Al-Cu-Mg precipitates, computes model energies, derives elemental references, and reports formation-energy, volume-per-atom, lattice, elastic, solute-solute, surface, stacking-fault, generalized stacking-fault, and solute-stacking-fault errors against the DFT reference data used by the legacy evalpot plotting workflow.

Data Provenance
---------------

The staged OQMD structures come from the Open Quantum Materials Database (OQMD), https://oqmd.org/, under the Creative Commons Attribution 4.0 International license. The staged special GSF layer structures are copied from the legacy evalpot data bundle for the Theta and Theta double-prime precipitate fault comparisons. Cite:

* Saal, Kirklin, Aykol, Meredig, and Wolverton, "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)", JOM 65, 1501-1509 (2013). doi:10.1007/s11837-013-0755-4
* Kirklin, Saal, Meredig, Thompson, Doak, Aykol, Ruhl, and Wolverton, "The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies", npj Computational Materials 1, 15010 (2015). doi:10.1038/npjcompumats.2015.10

Current Scope
-------------

The current slice validates the ML-PEG calculation, analysis, and app data flow for bulk crystals, elastic constants, solute-solute bindings, legacy FCC/HCP surface energies, relaxed FCC stable/unstable stacking-fault energies, relaxed Theta/Theta double-prime GSF curves, and relaxed solute-stacking-fault layer interactions. Elastic constants and solute-solute bindings are present as ``very_slow`` opt-in calculations because the legacy comparisons require many strained cells or relaxed FCC supercells. Antisite, cluster, triplet, and interface-energy workflows remain future work.
