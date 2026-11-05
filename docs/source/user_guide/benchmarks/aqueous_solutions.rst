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

* TODO: origin of the initial water box (who built it and how, or a citation).

Reference data:

* Ab initio molecular dynamics of the same system at the PBE-D3 level.
  TODO: code, settings, trajectory length and citation.
* RDF, VDOS and VACF analysis is adapted from:

  * C. Schran, F. L. Thiemann, P. Rowe, E. A. Müller, O. Marsalek, and A. Michaelides,
    "Machine learning potentials for complex aqueous systems made simple," Proc. Natl.
    Acad. Sci. U.S.A., vol. 118, e2110077118, Sept. 2021.
    https://doi.org/10.1073/pnas.2110077118


Ice
===

Summary
-------

Performance in reproducing the structure and dynamics of ice, compared to ab initio
molecular dynamics. The system is 96 water molecules (288 atoms) in a
13.49 x 15.58 x 14.64 Å orthorhombic cell. TODO: confirm the ice phase (e.g. ice Ih).

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

* TODO: origin of the initial ice structure (who built it and how, or a citation).

Reference data:

* Ab initio molecular dynamics of the same system at the PBE-D3 level.
  TODO: code, settings, trajectory length and citation.
* RDF, VDOS and VACF analysis is adapted from Schran et al. (2021), as for
  `Bulk water`_.
