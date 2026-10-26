"""Evaluate MLIPs on the Landscape17 DFT stationary points.

Landscape17 is a set of potential energy landscapes for six molecules, each
mapped as a network of minima connected by transition states, computed at
wB97X/6-31G(d). The reference contains 28 minima and 67 transition states, and
a DFT Hessian for every one of them.

This benchmark evaluates a model at those reference geometries and asks how
much of the landscape survives contact with it. Nothing here searches for new
structures; the model is handed the DFT answer and judged on what it does with
it. The companion ``landscape_search`` benchmark asks the harder question of
whether a model can find the landscape unaided.

The tests, in increasing order of what they demand of the model:

``test_singlepoints``
    Energies and forces at the reference geometries. The cheapest signal, and
    the one most comparable with other benchmarks.

``test_reconverge_minima``
    Relax each reference minimum under the model. A good model leaves it where
    it is; the distance it drifts is the metric.

``test_reconverge_transition_states``
    The same for saddles, by eigenvector-following. Harder, because a saddle is
    only stationary in one direction, and a model may have no saddle nearby at
    all.

``test_connectivity``
    Follow the steepest-descent paths from each reconverged saddle and check
    which minima they reach. A model can place a saddle correctly and still
    connect it to the wrong basins, which changes the kinetics while leaving
    every structure looking right.

``test_hessians``
    Compare the model's Hessian at each reference geometry with the DFT one:
    the overlap of the normal modes, the overlap of the reaction coordinate at
    saddles, and whether the curvature is systematically scaled.

Reported distributions rather than pass rates
---------------------------------------------
The drift and overlap tests report positions in a distribution -- medians and
upper percentiles -- rather than the fraction falling below a threshold. A
fraction is brittle in a way that misleads: a model that reproduces every
structure with the same small systematic displacement scores zero if that
displacement sits a hair outside the cut, which reads as total failure rather
than as the tidy, correctable offset it is. Percentiles say where the
distribution sits and whether it has a tail, and both can be given
``good``/``bad`` thresholds with a physical justification.

Status
------
Skeleton. The structure, metric names and data flow are defined; the
calculations are not yet implemented.
"""

from __future__ import annotations

raise NotImplementedError(
    "Skeleton only: see the module docstring for the intended tests."
)
