============
Interstitial
============

FE1SIA
======

Summary
-------

This benchmark evaluates the formation energies of a single self-interstitial atom (SIA) in a host lattice. The test includes formation energies for distinct configurations within a supercell.

Metrics
-------

1. RMSD

Root Mean Square Deviation of formation energies compared to DFT data.

The formation energy of a configuration :math:`E_f` is calculated as:

.. math::
    E_f = E_{config} - \frac{N_{config}}{N_{bulk}} E_{bulk}

where :math:`E_{config}` is the total energy of the interstitial configuration containing :math:`N_{config}` atoms, and :math:`E_{bulk}` is the energy of the perfect bulk supercell consisting of :math:`N_{bulk}` atoms.

The reference formation energies are derived from DFT calculations provided in the dataset.

Computational cost
------------------

Low: The geometries are static, requiring only single-point energy calculations for the configurations and the bulk reference.

Data availability
-----------------

Input structures:

* Subset ``formation_energy`` of the DFT dataset.

  * A. Allera, A. M. Goryaeva, P. Lafourcade, J.-B. Maillet, and M.-C. Marinica,
    Neighbors map: An efficient atomic descriptor for structural analysis,
    Computational Materials Science 231 (2024).

Reference data:

* Computed from the subset ``formation_energy`` of the DFT dataset as input structures.


Relastab
========

Summary
-------

This benchmark evaluates the ability of models to correctly rank the stability of different interstitial configurations. It focuses on the relative energy ordering of distinct interstitial structures in the dataset.

Metrics
-------

1. Kendall Tau

Kendall rank correlation coefficient (:math:`\tau`): a measure of rank correlation that evaluates the similarity of the orderings of the data. It assesses the number of *concordant* and *discordant* pairs of observations. For every pair of configurations, it checks if the model agrees with the reference on which is more stable.
A value of 1.0 indicates perfect agreement, 0.0 indicates no correlation, and -1.0 indicates perfect inversion.
This metric is sensitive to **pairwise ordering** errors. It is particularly robust for small datasets and focuses strictly on whether the relative stability order is preserved.

2. Spearman

Spearman rank correlation coefficient (:math:`\rho`): a non-parametric measure of rank correlation.
It is defined as the Pearson correlation coefficient between the *rank variables*. It converts raw energies to integer ranks and calculates the linear correlation between them.
Like Kendall Tau, values range from -1 to 1. An absolute value of 1 indicates a perfect monotonic function.
While both metrics evaluate ranking, Spearman assesses the general **monotonic relationship**, while Kendall Tau assesses the probability of correct pairwise discrimination.

Computational cost
------------------

Low: Requires single-point energy calculations for each configuration in the dataset.

Data availability
-----------------

Input structures:

* Subset ``relative_stability`` of the DFT dataset.

  * A. Allera, A. M. Goryaeva, P. Lafourcade, J.-B. Maillet, and M.-C. Marinica,
    Neighbors map: An efficient atomic descriptor for structural analysis,
    Computational Materials Science 231 (2024).

Reference data:

* Computed from the subset ``relative_stability`` of the DFT dataset as input structures.
