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
