==========
Landscapes
==========

Benchmarks built on Landscape17, a set of potential energy conformational landscapes
for six rMD17 molecules computed at ωB97X/6-31G(d). Each landscape is a network
of minima connected by transition states: 28 minima and 67 transition states in total,
with a DFT Hessian for every stationary point.

Every number in the tables is pooled over the six molecules, and so leans
heavily on aspirin: aspirin alone is 37 of the 67 transition states and 6,944
of the 11,173 path frames.

Every benchmark here runs the model in float64. That is a requirement rather than a
preference: eigenvector following takes its curvature from finite differences of
forces, which in float32 is noise-dominated, and the search wanders.

Every benchmark here needs topsearch, from the ``topsearch`` extra
(``uv sync --extra topsearch``), and is skipped without it.

.. _stationary-points:

Stationary points
=================

Summary
-------

The model is handed the DFT stationary points and judged on what it does with
them. Nothing is searched for: every geometry is supplied.

Metrics
-------

1. Energies and forces along the steepest-descent paths

   The 11,173 frames joining the stationary points are ordinary points on the
   surface with non-zero forces, so this is a conventional accuracy test -- but
   sampled along the reaction coordinates rather than from a thermal
   distribution, which is where a landscape benchmark should want to look.

2. Energy and force at each stationary point

   The energy error at the 95 DFT minima and transition states, each method
   measured from its own per-molecule mean so that a constant offset cancels.

   DFT forces are zero there by construction, so there is no reference to
   compare against. What the model reports is the residual: how far from
   stationary the DFT geometry is under this model, as an RMSE over every
   force component, pooled the same way as along the paths. It predicts
   whether the structure will survive reconvergence, and it is not
   interchangeable with the RMSD on reconvergence below -- a model can
   disagree sharply along a stiff direction and barely move, or mildly along
   a soft one and travel far.

3. RMSD on reconvergence

   Each reference minimum is relaxed under the model and each transition state
   refined by eigenvector following. A good model leaves the structure where it
   is; the distance it travels is the metric. Saddles are the harder case,
   having one direction of negative curvature, and a model may have no saddle
   nearby at all. Saddles converged is the fraction of the 67 for which
   eigenvector following reaches a saddle within its step budget, and the
   saddle distances are taken over those alone.

4. Connectivity preserved

   The fraction of all 67 reference transition states whose reconverged saddle
   still joins the same two minima as in DFT, so a saddle that fails to
   reconverge counts against it. This is reported with the saddle
   reconvergence rather than as a test of its own, because eigenvector
   following already returns the minima its steepest-descent paths reach;
   separating them would mean running the whole search twice for a result
   already in hand.

5. Hessian comparison

   Normal mode overlaps against the DFT Hessian, the overlap of the reaction
   mode at saddles, and how much softer or stiffer than DFT the model is
   overall, in percent. It also reports the curvature error over the softest
   third of the modes, where a landscape lives, and the ratio of the model's
   negative eigenvalue at a saddle to DFT's: the curvature along the reaction
   coordinate, a proxy for the imaginary frequency that enters the tunnelling
   correction and the transition state theory prefactor. That ratio is scored as
   ``min(r, 1/r)``, so a barrier top too sharp counts against a model as much
   as one too flat by the same factor. Reaction modes present is the fraction of
   the 67 saddles where the model, at the DFT geometry, has negative curvature
   along the mode matched to DFT's reaction mode; the reaction mode overlap and
   curvature are taken over those saddles alone. It is a snapshot at the DFT
   geometry, unlike Saddles converged, which counts searches that walk to a
   saddle of the model's own, and the two can differ widely.

   The model's Hessian is taken by central finite differences of its forces,
   with a 1e-3 Å step, at the DFT geometry. Both Hessians are compared with
   translations and rotations projected out, and modes are matched one to one
   by overlap. The Hessians are Cartesian, not mass-weighted, so their
   eigenvalues are curvatures rather than squared vibrational frequencies.

Scoring
-------

Path energy RMSE, Path force RMSE, the stationary-point Force RMSE and
Reaction modes present are reported but not scored. The path errors track the
stationary-point ones almost exactly, so scoring both would count the same
thing twice. Force RMSE mostly measures the gap between levels of theory rather
than the model: the reference geometries are not stationary at other levels,
and ωB97M-V/def2-TZVPD, the OMol25 level of theory, itself has an RMS force of
129 meV/Å at them. Reaction modes present says how many saddles the two
reaction mode metrics describe; the scored coverage is Saddles converged.

A model whose outputs of one kind are incomplete -- a structure missing, or a
file that cannot be read -- scores NaN on every metric built on them, rather
than being scored on the structures it has.

The thresholds come from the reference wherever it supplies one:

* **Energies.** *Bad* is what a model that knows nothing scores -- the same
  energy for every structure of a molecule -- which is the spread of the
  reference's own relative energies: 13.8 meV/atom at the stationary points and
  11.6 along the paths. *Good* is the reference uncertainty: scored against this
  reference, ωB97M-V/def2-TZVPD gets 1.98 meV/atom at the stationary points and
  1.51 on a 300-frame sample of the paths, so the thresholds are 2.0 and 1.5.
* **Forces** (unscored, so the thresholds only colour the columns). *Good* is
  the same measured gap, 129 meV/Å at the stationary points and 123 along the
  paths. *Bad* is 153 meV/Å, what predicting zero force along the paths scores.
* **Drift** (minima and saddle RMSD, median and 90th percentile). *Bad* is
  0.34 Å, the separation of the closest two distinct reference minima: past it,
  a structure that drifted cannot be told from one that landed on another
  minimum. *Good* is 0.01 Å, about the run-to-run noise of the saddle search on
  GPU, which leaves room for small geometry differences between functionals.
* **Coverage** (saddles converged, connectivity preserved). A perfect model of
  the reference scores 1 by construction, since every search starts at its own
  saddle. *Good* is 0.95 and 0.9, a judgement call leaving room for a model
  trained on another functional, whose surface may have lost or moved a few of
  these saddles. *Bad* is 0, what a model that finds nothing scores.
* **Reaction modes present** (unscored). The thresholds of Saddles converged:
  DFT has a reaction mode at all 67 saddles, so a perfect model scores 1, and
  one with none scores 0.
* **Mode overlaps.** 0.99 and 0.8, an 8° and a 37° misalignment, well above the
  0.32 that randomly oriented modes score.
* **Stiffness.** 2% and 20%. An error that scales the whole Hessian scales every
  frequency by its square root, so these are about 1% and 10% in frequency;
  10% is the error of uncorrected Hartree-Fock frequencies.
* **Soft mode curvature.** 0.02 and 0.2 eV/Å², about 1% and 7% of the median
  curvature of the softest third of the reference modes, 2.7 eV/Å².
* **Reaction mode curvature.** *Good* is 1. *Bad* is 0.44, what a single
  imaginary eigenvalue predicted for every saddle scores, so a model at or
  below it shows no skill there.

The ωB97M-V/def2-TZVPD figures are energies and analytic forces at the
reference geometries (gpu4pyscf, direct SCF, (99, 590) grid, (50, 194) VV10
grid), scored exactly as a model is.

Protocol
--------

The settings are fixed in ``calc_stationary_points.py``:

* Minima are relaxed with L-BFGS-B to a gradient criterion of 1e-6 eV/Å,
  within 1000 iterations; it usually stops just short of that, once the
  energy no longer decreases at machine precision, leaving forces of at most
  3e-5 eV/Å. A relaxation that scipy does not report as converged, or whose
  end point topsearch's curvature check does not accept as a minimum, has
  not arrived, and contributes NaN.
* Transition states are refined by hybrid eigenvector following (topsearch)
  within 150 steps, with a gradient criterion of 1e-2 eV/Å. Each endpoint of
  its steepest-descent paths is relaxed as the minima are, and has reached a
  minimum when no atom has more than 1e-2 eV/Å of force left and topsearch's
  curvature check accepts it as one; it is identified with the nearest
  reference minimum within 0.3 Å, and connectivity is credited only when both
  endpoints reach minima. A saddle not reached within the budget counts against
  Saddles converged and Connectivity preserved; some reference saddles never
  converge even given 3000 steps.
* Every structure is a neutral singlet.
* No dispersion correction is added, as the reference, ωB97X/6-31G(d), has
  none.

A model error on a structure is warned about and recorded rather than stopping
the run: it voids that model's metrics that depend on the structure, or, for a
saddle, counts as one not reached.

Benchmark speed
---------------

Slow: about 2.5 to 5 hours per model on one NVIDIA H200 for the MACE models
tested, almost all of it the saddle reconvergence (mace-mp-0a: 4.6 hours for the
saddles, half an hour for everything else).

Data availability
-----------------

Input structures:

* The 28 minima, 67 transition states and 11,173 steepest-descent path frames
  of Landscape17: Cărare, V. et al. Global properties of the energy landscape:
  a testing and training arena for machine learned potentials. npj Comput.
  Mater. (2025). https://doi.org/10.1038/s41524-025-01878-x

Reference data:

* Energies at the stationary points, and energies and forces along the paths:
  Landscape17, ωB97X/6-31G(d) in Psi4.
* Hessians at the 95 stationary points: computed for this benchmark in PySCF at
  the same level -- ωB97X/6-31G(d) with Cartesian d functions, restricted
  Kohn-Sham, (100, 770) grid, SCF converged to 1e-11. They are analytic,
  agree with finite differences (0.005 Å step) to within 0.4% on ethanol and
  malonaldehyde, and reproduce each Landscape17 energy to within 0.05 kJ/mol.

.. _landscape-search:

Landscape search
================

Summary
-------

The harder question: given one conformer and no further reference
information, how much of the landscape does the model find on its own, and how
accurately?

A model can hold every stationary point it is handed and still fail to locate
them unaided, and it can also invent minima the reference does not contain.

This tests the model and the search together, which is the point -- a potential
is only useful for exploring a surface if a search can traverse it -- but it
means the search settings form part of the benchmark. They are fixed, and the
same for every model.

Metrics
-------

A found structure counts only if the model holds it as what the search says it
is, judged on the model's Hessian with translation and rotation projected out.
A minimum must have no atom with more than 0.01 eV/Å of force left and no
curvature below -0.001 eV/Å². A transition state must be a first-order saddle:
its lowest curvature is below -0.001 eV/Å², and any second negative curvature
is at most a tenth of it -- the rule Landscape17 itself was built with, under
which two negative modes are one saddle when one dominates. The search itself
checks neither: topsearch ends a saddle search on the force alone, and adds the
minima its steepest descents end in without checking them. Its own check for a
minimum is not used, as it keeps translation and rotation in and passes a
second-order saddle whose forces are small.

The five searches find most structures several times, so copies are merged and
each structure counts once. Two found structures are copies when they are
closer than 0.15 Å for minima or 0.09 Å for saddles: just under half the gap
between the two closest distinct DFT structures of each kind (paracetamol's
minima 2 and 3, 0.340 Å apart; aspirin's two closest saddles, 0.186 Å apart),
so that a model that finds both of such a pair is not taken to have found one.
A merged structure is compared with DFT through its closest copy, and a
structure the model does not hold is never merged with ones it does.

1. Minima and transition states recovered

   The fraction of reference stationary points the search locates, within
   0.3 Å RMSD (permutation- and inversion-aware). Matching is one to one, so
   it counts the reference structures that have a match: two found minima
   near the same DFT minimum recover it once. The pairing makes as many
   matches as it can, then picks the closest.

2. RMSD to the matched reference structure

   How accurately the recovered structures are placed, over the matched pairs
   only and so always within the 0.3 Å matching radius.

3. Connectivity preserved

   The fraction of all 67 reference transition states that the search both
   recovered and found joining the same pair of minima as in the reference,
   through any of its copies of the saddle. Taken over every reference rather
   than the recovered ones, so a model that finds few cannot score well on the
   few it finds. Both endpoints must be minima the model holds as such, within
   0.3 Å of the reference pair, and two different minima of the search: a
   saddle whose steepest descents end in the same structure joins nothing.
   Where the reference records a saddle joining a minimum to itself, a
   rearrangement between symmetry-equivalent copies, the model is credited
   when its saddle joins two distinct copies of that minimum.

4. Extra minima and transition states

   Structures with the molecule's bonding, held by the model, further than
   0.3 Å from every reference structure, per reference structure. Divided by the reference
   count rather than by what the model found, so that finding more cannot
   dilute it. The DFT minima are taken as complete, so extra minima are scored:
   a model that adds them has a landscape DFT does not. Extra transition states
   are not scored, since the reference landscape was built with a narrower
   search than this one and some of them may be genuine DFT saddles it lacks.

5. Off-reference minima

   Minima whose bonding differs from the molecule's. The search only sets out
   to change conformation, with basin-hopping steps that rotate dihedral
   angles, so in an ideal case the bonding framework never changes. These are
   not automatically wrong -- a model may find a genuine isomer and only place
   it at the wrong energy -- but a model producing many of them lets bonds
   break or form where only the conformation should change.

6. Distorted minima

   Minima with the molecule's bonding but one of its bonds more than 20% away
   from its mean length over the DFT minima. The off-reference test compares
   bonding graphs and cannot see these: a bond crushed to two thirds of its
   length is still a bond, so the structure reads as the right molecule at an
   impossible geometry. The flagged structures are written out beside the
   model's results.

7. Unsettled minima and saddles not first-order

   The fractions of what the search reported as minima and as transition
   states that the model does not hold as such, as defined above. They are left
   out of every other metric; the off-reference and distorted fractions are of
   the minima the model holds.

8. RMSD to the nearest found structure

   For every reference minimum and transition state, the distance to the
   closest structure the model holds, matched or not; a reference with nothing
   found near it counts as 10 Å, past any real distance. Unlike item 2 this does not condition on success, so it does not
   flatter a model that found two structures precisely and missed the rest.

9. Minima near-missed

   Reference minima whose closest found structure lies between the 0.3 Å
   matching radius and 0.45 Å. Recovery is a pass rate over a hard radius, so
   a reference missed by 0.31 Å scores the same as one missed by 3 Å; this
   separates the two. Low recovery with many near misses means the model is
   systematically displaced rather than failing to find the structures, which
   is a correctable offset.

Scoring
-------

The score weights saddles recovered, connectivity and the two nearest-structure
distances by 1, minima recovered and distorted minima by 0.5, and extra minima
by 0.25; the remaining columns are shown but not scored. Distorted and extra
minima weigh less than the coverage rows because they measure what a model got
wrong among what it found, and a model that finds little has little to get
wrong: at equal weight, a model whose searches all crashed would outscore one
that ran and found a fair share of the landscape. Extra minima weigh least, as
how many a search turns up depends on how hard it looks. A model whose outputs are
incomplete -- a search missing, one the protocol does not run, a file that
cannot be read, or found structures the check step has not judged -- scores
NaN on every metric, rather than being scored on the searches that happen to
exist. A search that raised counts as one that found nothing.

The thresholds:

* **Coverage** (minima and saddles recovered, connectivity). *Good* is 1, the
  whole reference, and *bad* is 0, what a model that finds nothing scores. No
  model reaches 1 here: the search has a fixed budget, the same for every
  model, and a model that agreed with DFT exactly would still miss reference
  saddles that the search does not propose.
* **Distances** (nearest and matched RMSD). As for drift in the stationary
  points benchmark: *bad* is 0.34 Å, the separation of the closest two
  distinct reference minima, and *good* 0.01 Å. The matched RMSD cannot exceed
  the 0.3 Å matching radius, which is its *bad*.
* **Extra structures.** *Good* is 0. *Bad* is 0.5, half as many extra
  structures as the reference holds, a judgement call.
* **Bonding and shape** (off-reference, distorted, unsettled, not first-order,
  near-missed). *Good* is 0, which every DFT structure satisfies, and *bad*
  0.2, one in five, a judgement call.

Protocol
--------

The settings are fixed in ``calc_landscape_search.py``:

* Each molecule is searched five times, each from a different rMD17 conformer
  with its own random seed, by topsearch's network sampling: basin hopping for
  minima, then each minimum paired with its six nearest and the pairs
  connected by nudged elastic band and hybrid eigenvector following.
* Basin hopping takes 100 steps, each perturbing up to two dihedral angles by
  up to 180°; each minimisation runs L-BFGS-B to a gradient criterion of
  1e-3 eV/Å within 1000 iterations. Structures within 0.3 Å and
  5 meV of each other are one, but symmetry copies are kept apart so that a
  rearrangement between copies can be searched for, which departs from the
  published runs.
* Six neighbours is the furthest a reference saddle sits down the neighbour
  list of the minima it joins. Because symmetry copies are kept, and topsearch
  ranks neighbours by a distance under which a copy is zero away, copies take
  some of the six places: in mace-omol's searches the six nearest hold 2.4
  distinct structures on average in aspirin and 1.3 in paracetamol. Coverage
  is limited accordingly, equally for every model.
* Each pair gets up to three connection attempts. Eigenvector following stops
  when no gradient component exceeds 1e-2 eV/Å, within 100 steps, and descends
  from the saddle to its two minima with L-BFGS to 1e-3 eV/Å.
* After the search, a check step evaluates every found structure with the same
  model, without moving it: the largest force on any atom, and a
  central-difference Hessian (1e-3 Å steps). A structure the model could not
  evaluate is evaluated again the next time the step runs, as is any whose
  found file has changed since.
* Every structure is a neutral singlet, and no dispersion correction is added,
  as the reference has none.

A search that raises is recorded as failed rather than stopping the run, and is
retried the next time the calculation runs. Every output is written whole or
not at all, so a run that is stopped part-way can be restarted, and repeats only
the searches that had not finished.

Benchmark speed
---------------

Multi-day: one search takes from ten minutes to six and a half hours, and the
thirty of one model add up to 46 to 59 hours on an NVIDIA H200 for the four
MACE models measured, with four searches sharing each GPU. The searches are
independent, so they can be spread over GPUs. The check step adds roughly one
to five hours per model; it is marked multi-day with the search it judges, so
that one option runs both.

Data availability
-----------------

Input structures:

* Starting conformers: twenty frames per molecule, evenly spaced through each
  rMD17 trajectory and reordered to Landscape17's atom ordering; the first five
  are used. Christensen, A. S. & von Lilienfeld, O. A. On the role of gradients
  for machine learning of molecular energies and forces. Mach. Learn.: Sci.
  Technol. 1, 045018 (2020). https://doi.org/10.1088/2632-2153/abba6f

Reference data:

* The 28 minima and 67 transition states of Landscape17, ωB97X/6-31G(d) in
  Psi4: Cărare, V. et al. Global properties of the energy landscape: a testing
  and training arena for machine learned potentials. npj Comput. Mater. (2025).
  https://doi.org/10.1038/s41524-025-01878-x

Two of the reference saddles are recorded as joining a minimum to itself while
their own DFT steepest-descent paths join two different minima: aspirin
ts4_4_0 joins minima 10 and 4, and ts6_6_0 minima 6 and 9. No model can be
credited with their connectivity as recorded, in this benchmark or the
stationary points one, which caps connectivity at 65 of 67 for every model,
DFT included. A corrected release of Landscape17 will follow.
