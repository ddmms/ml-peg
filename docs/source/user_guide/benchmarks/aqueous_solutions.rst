==================
Aqueous solutions
==================

Bulk water
==========

Summary
-------

Performance in reproducing the structure and dynamics of liquid water, compared to
ab initio molecular dynamics. The system is 64 water molecules (192 atoms) in a
12.42 Å cubic box.

Metrics
-------

Each model is used to run Langevin NVT molecular dynamics at 330 K with a 0.5 fs
timestep. The first 25 ps are discarded as equilibration, and the following 300 ps are
used for analysis, with frames stored every 2 fs. A D3(BJ) dispersion correction is
added to models that were not trained with dispersion.

All three metrics are curve similarity scores. For each curve ``f`` (e.g. one element
pair's RDF), the model curve is interpolated onto the reference grid and the score is
``100 * (1 - sum|f_ref - f_model| / (sum f_ref + sum f_model))``, so 100 corresponds to
perfect agreement. Each metric is the mean score over all curves of that type.

1. RDF score

Radial distribution functions for the O-O, O-H and H-H pairs, computed up to half the
box length.

2. VDOS score

Vibrational density of states for O and H, obtained from the Fourier transform of the
velocity autocorrelation function.

3. VACF score

Velocity autocorrelation function for O and H, normalised to one at zero time. As VACFs
can be negative, absolute values are used in the denominator of the score.

Benchmark speed
---------------

Very slow: 650,000 MD steps (325 ps at a 0.5 fs timestep) of a 192-atom system.
TODO: rough runtime for a representative model, e.g. ``mace-mp-0a``, on named GPU.

Data availability
-----------------

Input data:

* The initial configuration is a periodic box of liquid water at the experimental
  density under ambient conditions, containing 64 water molecules in a 12.42 Å cubic
  cell.

Reference data:

* A 100 ps ab initio molecular dynamics trajectory of the same system, generated
  with CP2K at the PBE-D3(0) level (zero-damping D3 dispersion) and used in the
  MACE-MP-0 study [1]_.
* The curve similarity metrics also follow the scoring framework of Advincula
  et al. [2]_.
* RDF, VDOS and VACF analysis is adapted from:

  * C. Schran, F. L. Thiemann, P. Rowe, E. A. Müller, O. Marsalek, and A. Michaelides,
    "Machine learning potentials for complex aqueous systems made simple," Proc. Natl.
    Acad. Sci. U.S.A., vol. 118, e2110077118, Sept. 2021.
    https://doi.org/10.1073/pnas.2110077118


Ice
===

Summary
-------

Performance in reproducing the structure and dynamics of hexagonal ice Ih, compared to ab initio
molecular dynamics. The system is 96 water molecules (288 atoms) in a
13.49 x 15.58 x 14.64 Å orthorhombic cell.

Metrics
-------

Each model is used to run Langevin NVT molecular dynamics at 250 K with a 0.5 fs
timestep. The first 25 ps are discarded as equilibration, and the following 300 ps are
used for analysis, with frames stored every 2 fs. A D3(BJ) dispersion correction is
added to models that were not trained with dispersion.

The metrics are the same curve similarity scores as for `Bulk water`_:

1. RDF score

Radial distribution functions for the O-O, O-H and H-H pairs, computed up to half the
shortest cell length.

2. VDOS score

Vibrational density of states for O and H.

3. VACF score

Velocity autocorrelation function for O and H, normalised to one at zero time.

Benchmark speed
---------------

Very slow: 650,000 MD steps (325 ps at a 0.5 fs timestep) of a 288-atom system.
TODO: rough runtime for a representative model, e.g. ``mace-mp-0a``, on named GPU.

Data availability
-----------------

Input data:

* A supplied ice Ih configuration containing 96 water molecules in an orthorhombic
  simulation cell. Its original construction procedure and source have not been
  established.

Reference data:

* A 100 ps ab initio molecular dynamics trajectory of the same system, generated
  with CP2K at the PBE-D3(0) level (zero-damping D3 dispersion) and used in the
  MACE-MP-0 study [1]_.
* RDF, VDOS and VACF analysis is adapted from Schran et al. (2021), with the
  curve similarity metrics also following Advincula et al. [2]_, as for
  `Bulk water`_.

.. [1] I. Batatia et al., "A foundation model for atomistic materials chemistry,"
   The Journal of Chemical Physics **163**, 184110 (2025).
   https://doi.org/10.1063/5.0297006

.. [2] X. R. Advincula, Y. Litman, J. Shi, F. Della Pia, C. Schran, and
   A. Michaelides, "How Good Is DFT for Solid-Liquid Interfaces? A Comparison
   With the Random-Phase Approximation for Water on Graphene,"
   arXiv:2610.09685 (2026). https://arxiv.org/abs/2610.09685
