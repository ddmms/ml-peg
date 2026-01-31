========
Surfaces
========

OC157
=====

Summary
-------

Performance in predicting relative energies between three configurations for 157
molecule-surface combinations.

Metrics
-------

1. Energy error

How accurate all relatve energy predictions are.

For each group of three structures, the relative energies are calculated for all pairs
of structures. Models receive a score based on the mean difference between these
predictions and the reference, averaged over all pairs and over all combinations.

2. Ranking error

Whether the most and least stable strucutres are predicted.

For each group of three structures, the relative energies are calculated for all pairs
of structures. Models receive a score of 0, 0.5, or 1, based on whether the predicted
lowest and highest energy pairs match the reference predictions, and this is averaged
for all 157 combinations.

Computational cost
------------------

Low: tests are likely to take a couple of minutes to run on CPU.

Data availability
-----------------

Input data:

* Surfaces were taken from the Open Catalyst Challenge 2023

  * L. Chanussot, A. Das, S. Goyal, T. Lavril, M. Shuaibi, M. Riviere, K. Tran, J. Heras-Domingo, C. Ho, W. Hu, A. Palizhati, A. Sriram, B. Wood, J. Yoon, D. Parikh, C. L. Zitnick, and Z. Ulissi, “Open Catalyst 2020 (OC20) dataset and community challenges,” ACS Catal., vol. 11, pp. 6059–6072, May 2021.
  * R. Tran, J. Lan, M. Shuaibi, B. M. Wood, S. Goyal, A. Das, J. Heras-Domingo, A. Kolluru, A. Rizvi, N. Shoghi, A. Sriram, F. Therrien, J. Abed, O. Voznyy, E. H. Sargent, Z. Ulissi, and C. L. Zitnick, “The Open Catalyst 2022 (OC22) data set and challenges for oxide electro catalysts,” ACS Catal., vol.13, pp. 3066–3084, Mar. 2023.

* Structures containing oxygen (O) and several transition metals (Co, Cr, Fe, Mn, Mo,
  Ni, V and W) were exlcuded due to Hubbard U correction

Reference data:

* Same as input data
* PBE-D3(BJ), MPRelaxSet settings


S24
===

Summary
-------

Performance in predicting adsorption energies for a diverse set of surfaces and adsorbates.

Metrics
-------

Adsorption energy error

For each combination of surface, molecule, and surface + molecule, the adsorption
energy is calculated by taking the difference between the energy of the surface +
molecule and the sum of individual surface and molecule energies. This is compared to
the reference adsorption energy, calculated in the same way.

Computational cost
------------------

Very low: tests are likely to take less than a minute to run on CPU.

Data availability
-----------------

Input data:

* Structures were taken from an amalgamation of published and unpublished works, including:

  * Y. S. Al-Hamdani, M. Rossi, D. Alfè, T. Tsatsoulis, B. Ramberger, J. G. Brandenburg, A. Zen, G. Kresse, A. Grüneis, A. Tkatchenko, and A. Michaelides, “Properties of the water to boron nitride interaction: From zero to two dimensions with benchmark accuracy,” J. Chem. Phys., vol. 147, p. 044710, July 2017. 35
  * J. G. Brandenburg, A. Zen, M. Fitzner, B. Ramberger, G. Kresse, T. Tsatsoulis, A. Grüneis, A. Michaelides, and D. Alfè, “Physisorption of water on graphene: Subchemical accuracy from many- body electronic structure methods,” J. Phys. Chem. Lett., vol. 10, pp. 358–368, Feb. 2019.
  * C. Ehlert, A. Piras, and G. Gryn’ova, “CO2 on graphene: Benchmarking computational approaches to noncovalent interactions,” ACS Omega, vol. 8, pp. 35768–35778, Oct. 2023.
  * T. Tsatsoulis, S. Sakong, A. Groß, and A. Grüneis, “Reaction energetics of hydrogen on Si(100) surface: A periodic many-electron theory study,” J. Chem. Phys., vol. 149, p. 244105, Dec. 2018.
  * T. Tsatsoulis, F. Hummel, D. Usvyat, M. Schütz, G. H. Booth, S. S. Binnie, M. J. Gillan, D. Alfè, A. Michaelides, and A. Grüneis, “A comparison between quantum chemistry and quantum Monte Carlo techniques for the adsorption of water on the (001) LiH surface,” J. Chem. Phys., vol. 146, p. 204108, May 2017.
  *  H.-Z. Ye and T. C. Berkelbach, “Ab initio surface chemistry with chemical accuracy,” arXiv preprint arXiv:2309.14640, 2023.
  * P. G. Lustemberg, P. N. Plessow, Y. Wang, C. Yang, A. Nefedov, F. Studt, C. Wöll, and M. V. Ganduglia-Pirovano, “Vibrational frequencies of cerium-oxide-bound CO: A challenge for conventional dft methods,” Phys. Rev. Lett., vol. 125, p. 256101, Dec. 2020.
  * B. X. Shi, A. Zen, V. Kapil, P. R. Nagy, A. Grüneis, and A. Michaelides, “Many-body methods for surface chemistry come of age: Achieving consensus with experiments,” J. Am. Chem. Soc., vol. 145, pp. 25372–25381, Nov. 2023.
  *  N. Hanikel, X. Pei, S. Chheda, H. Lyu, W. Jeong, J. Sauer, L. Gagliardi, and O. M. Yaghi, “Evolution of water structures in metal-organic frameworks for improved atmospheric water harvesting,” Science, vol. 374, pp. 454–459, 2021.
  * F. Berger, M. Rybicki, and J. Sauer, “Molecular dynamics with chemical accuracy–Alkane adsorption in acidic zeolites,” ACS Catal., vol. 13, pp. 2011–2024, 2023.
  * F. Berger and J. Sauer, “Dimerization of linear butenes and pentenes in an acidic zeolite (H-MFI),” Angew. Chem., Int. Ed., vol. 60, pp. 3529–3533, 2021.

Reference data:

* Same as input data
* PBE-D3(BJ), MPRelaxSet settings


Elemental Slab Oxygen Adsorption
================================

Summary
-------

Performance in predicting adsorption energies for oxygen on elemental slabs.

Metrics
-------

Adsorption energy error

For each slab, two single points are performed.
The first is for the isolated slab and the second is the slab with
an oxygen placed on-top of the site furthest along the slab's normal
direction. The distance between the oxygen and the site is that which
minimizes the energy according to MACE-MatPES-r2SCAN.

Computational cost
------------------

Very low: tests are likely to take less than a minute to run on CPU.

Data availability
-----------------

Input data:

* Elemental slabs were obtained using the Materials Project API. The lowest-surface-energy slab
  of the most stable experimentally observed bulk crystal is chosen. The relaxed slabs were
  submitted to the Materials Project by:

  * R. Tran, Z. Xu, B. Radhakrishnan, D. Winston, W. Sun, K. A. Persson, S. P. Ong, "Surface Energies of Elemental Crystals," Scientific Data, 2016, 3:160080. https://doi.org/10.1038/sdata.2016.80

Reference data:

* PBE single points are performed with the MatPESStatic set, with a cutoff energy of 520 eV.

  * A. D. Kaplan, R. Liu, J. Qi, T. W. Ko, B. Deng, J. Riebesell, G. Ceder, K. A. Persson, S. P. Ong, "A Foundational Potential Energy Surface Dataset for Materials," arXiv preprint arXiv:2503.04070, 2025. https://doi.org/10.48550/arXiv.2503.04070
  * S. P. Ong, W. D. Richards, A. Jain, G. Hautier, M. Kocher, S. Cholia, D. Gunter, V. Chevrier, K. A. Persson, G. Ceder, "Python Materials Genomics (pymatgen): A Robust, Open-Source Python Library for Materials Analysis," Comput. Mater. Sci., 2013, 68, 314–319. https://doi.org/10.1016/j.commatsci.2012.10.028

* Tran et al. relaxed the slabs using spin-polarized PBE calculations performed in VASP, with a cutoff energy of 400 eV.


Graphene Wetting Under Strain
=============================

Summary
-------

Performance in predicting adsorption energies for a water molecule on graphene under varying strain conditions.

Metrics
-------

MAE of adsorption energies

For each combination of water molecule orientation, water-graphene distance, and strain
condition, the adsorption energy is calculated by taking the difference between the
energy of the combined water + graphene system and the sum of individual water and
graphene energies. This is compared to the reference adsorption energy, calculated in the
same way.

MAE of binding energies & lengths

The adsorption energies calculated above are fitted to Morse potentials, to obtain an
effective binding energy and binding length (i.e. minimum of adsorption energy curve) for
each strain condition. This is compared to the reference binding energy & length,
calculated in the same way.

Computational cost
------------------

Very low: tests are likely to take less than a minute to run on CPU.

Data availability
-----------------

Input data:

* Structures were taken from:

  * D. W. Lim, X. R. Advincula, W. C. Witt, F. L. Thiemann, C. Schran, “Revealing Strain Effects on the Graphene-Water Contact Angle Using a Machine Learning Potential,” *awaiting publication* (arXiv:2601.20134)

Reference data:

* Same as input data
* PBE (with D3 dispersion correction), FHI-aims "intermediate" settings
