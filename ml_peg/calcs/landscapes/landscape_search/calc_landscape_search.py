"""Build a Landscape17 landscape from scratch under an MLIP and compare it to DFT.

The companion ``stationary_points`` benchmark hands the model the DFT answer
and asks what it does with it. This one gives the model a single 
conformer and asks it to find the landscape itself: basin hopping for the
minima, then double-ended searches between selected pairs for the transition
states, then a comparison of the resulting network with the DFT one.

That makes this a test of the model and the search together, which is the point
-- a potential is only useful for exploring a surface if a search can actually
traverse it -- but it also means the search settings are part of the benchmark
and have to be justified rather than inherited.

``test_build_landscape``
    Basin hopping and connection attempts, several independent seeds merged.

``test_compare_to_reference``
    Match the model's minima and saddles to the DFT ones, one to one, and
    report how far apart the matched pairs are and how much of the reference
    was recovered at all. Also report connectivity.

Coverage and accuracy are different questions
---------------------------------------------
A model that finds few structures but places them perfectly, and one that finds
everything roughly, are not comparable on a single number. Accuracy is reported
as a distribution over the matched pairs, and coverage separately as the
fraction of the reference that was found; both as a function of search effort,
since neither is converged at small budgets.

Search settings are a reported quantity, not a hidden default
--------------------------------------------------------------
How many pairs of minima the search attempts to connect is the dominant control
on how much of the reference it can recover, far more than the number of seeds,
and the published default leaves most of the reference unreachable. The setting
used is recorded with the results so a number can never be read without it.

Mean first passage time
-----------------------
Once a network exists, the kinetics it implies can be compared with the
reference directly rather than through structural proxies. Not yet implemented;
this is the natural place for it.

Status
------
Skeleton. The structure, metric names and data flow are defined; the
calculations are not yet implemented.
"""

from __future__ import annotations

raise NotImplementedError(
    "Skeleton only: see the module docstring for the intended tests."
)
