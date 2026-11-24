======
Models
======

The machine-learned interatomic potentials (MLIPs) known to ML-PEG are listed
below. Each entry is generated from ``ml_peg/models/models.yml``, so it always
matches the configuration ML-PEG loads. Descriptions, and the families models
are grouped into, are defined in ``docs/source/user_guide/model_notes.yml``.

.. model-registry::

UMA molecular crystals
----------------------

``uma-s-1p1-omc`` and ``uma-s-1p2p1-omc`` select the OMC task of
UMA-S-1.1 and UMA-S-1.2.1 respectively (PBE+D3), listing the full
checkpoint training datasets.
Their registry entries above
link the checkpoint, licence, citation and precision details.

MACE cuEq acceleration
----------------------

On Linux with an NVIDIA GPU, install MACE with cuEq::

   uv sync --extra mace-cueq

In the model YAML, set ``device: "cuda"`` and add ``enable_cueq: true`` under
``kwargs``, preserving the existing checkpoint settings. Restart running
calculations to apply the change.


ShiftML (NMR shielding models)
==============================

These models predict NMR chemical shieldings, not energies and forces, and are used
only by the :doc:`NMR Models – Organic Materials </user_guide/benchmarks/nmr_organic_materials>`
benchmarks. They are configured in ``ml_peg/models/nmr_models.yml`` rather than
``models.yml``, and installed with ``uv sync --extra shiftml``. The model weights are
downloaded from Zenodo on first use and cached.

ShiftML3
--------

Trained on periodic GIPAW-PBE shielding tensors.

- **Citation:** Kellner et al. (2025).
  `A Deep Learning Model for Chemical Shieldings in Molecular Organic Solids Including Anisotropy <https://doi.org/10.1021/acs.jpclett.5c01819>`__.

.. code-block:: yaml

  ShiftML3:
    module: shiftml.ase
    class_name: ShiftML
    device: "auto"
    trained_on_dispersion: true
    level_of_theory: PBE
    datasets: null
    additional_supported_elements: [H, C, N, O, S, F, P, Cl, Na, Ca, Mg, K]
    kwargs:
      model_version: ShiftML3

ShiftML4
--------

Trained on GIPAW-PBE shielding tensors plus a PBE0-minus-PBE correction computed on
each isolated molecule.

- **Citation:** Kellner et al. (2026). arXiv:2608.21313.

.. code-block:: yaml

  ShiftML4:
    module: shiftml.ase
    class_name: ShiftML
    device: "auto"
    trained_on_dispersion: true
    level_of_theory: PBE+ΔPBE0(monomer)
    datasets: null
    additional_supported_elements: [H, C, N, O, S, F, P, Cl, Na, Ca, Mg, K]
    kwargs:
      model_version: ShiftML4
