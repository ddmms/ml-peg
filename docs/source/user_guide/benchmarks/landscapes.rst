==========
Landscapes
==========

Benchmarks built on Landscape17, a set of potential energy conformational landscapes 
for six rMD17 molecules computed at wB97X/6-31G(d). Each landscape is a network 
of minima connected by transition states: 28 minima and 67 transition states in total, 
with a DFT Hessian for every stationary point.

.. _stationary-points:

Stationary points
-----------------

The model is handed the DFT stationary points and judged on what it does with
them: energies and forces there, how far each structure drifts when reconverged 
under the model, whether its transition states still connect the same minima, 
and how its Hessians compare with the DFT ones.

Drift and overlap are reported as Root Mean Squared Deviation values in a distribution 
rather than as the fraction falling below a threshold. 

.. _landscape-search:

Landscape search
----------------

The harder question: given one conformer and no further reference
information, how much of the landscape does the model find on its own, and how
accurately?

This tests the model and the search together, which is the point -- a potential
is only useful for exploring a surface if a search can traverse it -- but it
means the search settings form part of the benchmark and are reported with the
results rather than left as defaults. All models use the same settings. 
