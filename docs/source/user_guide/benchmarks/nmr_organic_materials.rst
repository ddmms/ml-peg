==============================
NMR Models – Organic Materials
==============================

Benchmarks for machine-learned **NMR chemical-shielding models** of molecular
organic crystals. These models predict shieldings rather than energies and forces,
so this category is scored separately from the interatomic-potential categories and
does not contribute to the overall summary score.

The models are compared with experimental chemical shifts rather than with DFT,
because they are trained on different DFT references; a DFT reference would mostly
measure that intended difference. Experimental values are taken from the original
assignment papers and mapped atom by atom onto the structures. Where the literature
leaves an assignment ambiguous, it is resolved once against the GIPAW-PBE shieldings
and used identically for every model, so no model can pick its own most favourable
assignment.


NMR Isotropic Shifts
====================

Summary
-------

Performance in predicting isotropic ¹H, ¹³C and ¹⁵N chemical shifts of 49 molecular
crystals, compared with experiment. ShiftML3 is trained on periodic GIPAW-PBE
shieldings; ShiftML4 is trained on GIPAW-PBE shieldings plus a PBE0 molecular
correction.


Metrics
-------

The models predict shieldings :math:`\sigma`, which are converted to shifts with one
straight line per nucleus and model, fitted by weighted least squares over all sites of
all structures:

.. math::

   (a, b) = \arg\min_{a,\,b} \sum_i w_i \left(\delta^{\mathrm{exp}}_i - a\,\sigma_i - b\right)^2

where :math:`\delta^{\mathrm{exp}}_i` is the experimental shift of site
:math:`i`, :math:`\sigma_i` its predicted shielding, averaged over its atoms and their
symmetry copies in the cell, and :math:`w_i` its weight. The slope :math:`a` is fitted
rather than fixed at −1. The predicted shifts are then

.. math::

   \delta_i = a\,\sigma_i + b

The weight :math:`w_i` is the number of protons a ¹H site represents: protons that
exchange quickly on one atom and share one reported shift, such as those of a methyl
group, form one site of weight 3. All other sites, including all ¹³C and ¹⁵N sites, have
weight 1. Because the line is fitted to the same experimental shifts that the errors
below are measured on, the errors are slightly lower than they would be for new
structures (by under 1% for ¹H and ¹³C, and about 3% for ¹⁵N).

1. Isotropic shift Root Mean Square Error (RMSE) (¹H, ¹³C, ¹⁵N)

   The root-mean-square difference between predicted and experimental shifts, for
   each nucleus:

   .. math::

      \mathrm{RMSE} = \sqrt{\frac{\sum_i w_i \left(\delta^{\mathrm{exp}}_i - \delta_i\right)^2}{\sum_i w_i}}

2. Isotropic shift Mean Absolute Error (MAE) (¹H, ¹³C, ¹⁵N)

   The weighted mean absolute difference about the same line,

   .. math::

      \mathrm{MAE} = \frac{\sum_i w_i \left|\delta^{\mathrm{exp}}_i - \delta_i\right|}{\sum_i w_i}

   It is less influenced than the RMSE by a few large errors.


Computational cost
------------------

Low. Measured on 2026-11-04 with ``ml_peg calc --durations=0``, once on 4 cores of an
AMD EPYC 7452 CPU and once on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU
(96 GB; ``device: "auto"`` in ``nmr_models.yml`` picks the GPU when
there is one). Times per benchmark, including model load:

======== ====== ========= ====== ======= =======
Model    Device Isotropic Tensor Cocaine Total
======== ====== ========= ====== ======= =======
ShiftML3 CPU    153 s     59 s   155 s   6.1 min
ShiftML4 CPU    128 s     51 s   135 s   5.3 min
ShiftML3 GPU    18 s      13 s   14 s    45 s
ShiftML4 GPU    13 s      11 s   12 s    36 s
======== ====== ========= ====== ======= =======

Peak memory on the CPU is 1.0–1.4 GB per model and benchmark.

ShiftML4 is faster because its committee has 7 members to ShiftML3's 8. Times depend
on the hardware and thread count. The input data (0.25 MB) is downloaded once and
cached. ShiftML downloads its model weights (15 files of about 10 MB, from Zenodo) on
first use and caches them; on a slow connection this can take longer than the
calculations.


Data availability
-----------------

Input structures:

* M. Kellner, J. B. Holmes, R. Rodriguez-Madrid, F. Viscosi, Y. Zhang, L. Emsley and
  M. Ceriotti, A deep learning model for chemical shieldings in molecular organic
  solids including anisotropy, J. Phys. Chem. Lett. 16, 8714 (2025); data record:
  Materials Cloud Archive, DOI 10.24435/materialscloud:kt-kd (CC-BY-4.0).
* The ¹³C and ¹⁵N structures are those of S. A. Ramos, L. J. Mueller and G. J. O.
  Beran, Faraday Discuss. 255, 119 (2025), optimised with PBE-D3(BJ).

Experimental reference data:

* ¹H: E. Carignani et al., J. Phys. Chem. C 117, 17731 (2013); A.-C. Uldry et al.,
  J. Am. Chem. Soc. 130, 945 (2008); M. Sardo et al., Solid State Nucl. Magn. Reson.
  65, 49 (2015); M. Baias et al., Phys. Chem. Chem. Phys. 15, 8069 (2013); M. Baias et
  al., J. Am. Chem. Soc. 135, 17501 (2013); C. M. Widdifield, H. Robson and
  P. Hodgkinson, Chem. Commun. 52, 6685 (2016); K. Maruyoshi et al., Chem. Commun. 48,
  10844 (2012); N. Mifsud et al., Phys. Chem. Chem. Phys. 8, 3418 (2006); C. Gervais
  et al., Magn. Reson. Chem. 42, 445 (2004); M. Cordova et al., Nat. Commun. 12, 2964
  (2021).
* ¹³C and ¹⁵N: Ramos, Mueller & Beran (2025), ESI Tables S5 and S7.


NMR Shift Tensors
=================

Summary
-------

Performance in predicting the principal components of the ¹³C chemical-shift tensor for
118 carbon sites in 19 molecular crystals, compared with experiment. These are the
structures used in the ShiftML papers' tensor benchmark.


Metrics
-------

1. ¹³C tensor Root Mean Square Error (RMSE)
2. ¹³C tensor Mean Absolute Error (MAE)

As for isotropic shifts, but over the three principal components of each carbon's
tensor. Each predicted shielding tensor is symmetrised and diagonalised, and its
eigenvalues :math:`\sigma_{11} \le \sigma_{22} \le \sigma_{33}` are paired with the
experimental :math:`\delta_{11} \ge \delta_{22} \ge \delta_{33}`. One unweighted line
:math:`\delta = a\,\sigma + b` per model is fitted through all three components of all
sites, and the RMSE and MAE are taken over all components.


Computational cost
------------------

Low (21 structures, 1639 atoms; full shielding tensors). Run times on CPU and GPU are
given with the isotropic benchmark, in one table for all three benchmarks.

The input data (0.09 MB) is downloaded once and cached.


Data availability
-----------------

Input structures:

* The ¹³C structures of the isotropic benchmark (Kellner et al. 2025 data record).

Experimental reference data:

* J. D. Hartman, S. Monaco, B. Schatschneider and G. J. O. Beran, Fragment-based ¹³C
  nuclear magnetic resonance chemical shift predictions in molecular crystals: An
  alternative to planewave methods, J. Chem. Phys. 143, 102809 (2015), SI Table S5
  (original measurements cited therein). Two transposed pairs of rows (GLUTAM01,
  ADENOS12) are corrected, as noted in the input data.


Structure Identification - Cocaine
==================================

Summary
-------

Identifying the correct crystal structure of cocaine among 29 candidate structures from
crystal structure prediction, by comparing predicted and experimental ¹H shifts,
following the atomistic-cookbook recipe on NMR crystallography with ShiftML3. The
correct structure is the prediction that matches the X-ray structure. It covers a
single crystal, so it does not count towards the category score.


Metrics
-------

1. Identification margin

For each candidate, the ¹H shieldings of one molecule are grouped into the 17 assigned
sites (the recipe's "environments") and converted to shifts with the slope fixed at −1
and an intercept fitted per candidate, and the RMSE against experiment is computed. The
margin is the RMSE of the best wrong candidate minus that of the correct structure;
higher is better, and zero or below means the correct structure is not identified. The
rank of the correct structure and its RMSE are also shown, for information.


Computational cost
------------------

Low (29 structures, 4214 atoms). Run times on CPU and GPU are given with the isotropic
benchmark, in one table for all three benchmarks.


Data availability
-----------------

Input structures:

* 29 candidate structures with GIPAW-PBE shieldings, from ``ShiftML_poly.zip``: M.
  Cordova et al., J. Phys. Chem. C 126, 16710 (2022); Zenodo record 7097427
  (CC-BY-4.0).

Experimental reference data:

* M. Baias et al., Phys. Chem. Chem. Phys. 15, 8069 (2013), Table 1, as used in the
  atomistic-cookbook recipe, with the values of the C5 bridgehead proton and a C4
  proton restored to the original assignment.


Tool and computational resource disclosure
==========================================

The calculation, analysis and application code of these benchmarks, the shared NMR
helpers, the supporting changes to ml-peg's model loading and structure viewer, the
scripts used to prepare the input data, and this page were drafted with an AI coding
assistant (Claude Code, with Claude Opus 5.5, Anthropic). The implementer chose the
reference data and the scientific conventions, reviewed, edited and verified the code
and this page, wrote the unit tests, and takes responsibility for them. No new
reference calculations were run: the GIPAW-PBE shieldings and experimental shifts are
taken from the published sources listed under each benchmark, and the timings above
were measured on the hardware named under "Computational cost".
