=======
Defects
=======

Split vacancy
=============

Summary
-------

Performance predicting split vacancy formation energies and relaxed structures for
metal oxides (PBEsol, 531 host compounds, 722 material-cation pairs, 2154 structures)
and stable nitrides (PBE, 144 host compounds, 149 material-cation pairs, 285 structures).

A split vacancy is a stoichiometry-conserving defect complex in which an isolated atomic
vacancy reconstructs into two vacancies and an interstitial
(:math:`V_X \to [V_X + X_i + V_X]`), often with a dramatic energy lowering.

Data from Seán Kavanagh, *Identifying split vacancy defects with machine-learned foundation models and electrostatics*,
`https://doi.org/10.1088/2515-7655/ade916 <https://doi.org/10.1088/2515-7655/ade916>`_


Metrics
-------

1. MAE (formation energy)

Mean absolute error of the split vacancy formation energy, defined as the energy
difference between the lowest-energy relaxed split vacancy (SV) and normal vacancy (NV)
structures:

.. math::

   E_\text{form} = \min_i E^\text{SV}_i - \min_j E^\text{NV}_j

where the minima are taken over all initial structures that match the DFT reference
(see metric 3) for a given material-cation pair.

2. Spearman's rank correlation

`Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_
between MLIP and DFT total energies, evaluated as single points on DFT-relaxed NV and
SV structures. A perfect ranking gives a coefficient of 1. The mean across all
material-cation pairs is reported.

3. Match Rate

Fraction of MLIP-relaxed structures that converge to the same geometry as the
DFT-relaxed reference, determined using the pymatgen
`StructureMatcher <https://pymatgen.org/pymatgen.analysis.html#module-pymatgen.analysis.structure_matcher>`_.
The match criterion is a normalised maximum atomic displacement below 0.3 (see
metric 4).

4. Max Dist

Maximum atomic displacement between the MLIP-relaxed and DFT-relaxed matched
structures, normalised by :math:`(V/N)^{1/3}` (wher :math:`V` is the cell volume and
:math:`N` the number of atoms) to give a unitless quantity comparable across
different crystal structures. Only computed for structure pairs that pass the
StructureMatcher test. The match criterion itself is a normalised max dist below 0.3.


Computational cost
------------------

Relatively slow: relaxations involve large defect supercells (50–500 atoms) and
multiple initial structures per material-cation pair.


Data availability
-----------------

Input structures:

* Generated using the `doped <https://github.com/SMTG-Bham/doped>`_ supercell
  algorithm. For oxides, supercell parameters are consistent with those of
  Kumagai et al. (using the ``vise`` package). Supercells satisfy a minimum image
  distance of 10 Å and a minimum of 50 atoms.

Reference data:

* See: Seán Kavanagh, *Identifying split vacancy defects with machine-learned foundation models and electrostatics*,
  `https://doi.org/10.1088/2515-7655/ade916 <https://doi.org/10.1088/2515-7655/ade916>`_
* All DFT calculations performed with VASP using PAW pseudopotentials. Oxides:
  PBEsol functional, 400 eV plane-wave cutoff, Γ-point sampling, 0.01 eV Å⁻¹
  force convergence. Nitrides: PBE functional with MPRelaxSet parameters,
  520 eV plane-wave cutoff. See paper for full details.
