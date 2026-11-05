======
Models
======

The machine-learned interatomic potentials (MLIPs) known to ML-PEG are listed
below. Each entry is generated from ``ml_peg/models/models.yml``, so it always
matches the configuration ML-PEG loads. Descriptions, and the families models
are grouped into, are defined in ``docs/source/user_guide/model_notes.yml``.

.. model-registry::

MACE cuEq acceleration
----------------------

On Linux with an NVIDIA GPU, install MACE with cuEq::

   uv sync --extra mace-cueq

In the model YAML, set ``device: "cuda"`` and add ``enable_cueq: true`` under
``kwargs``, preserving the existing checkpoint settings. Restart running
calculations to apply the change.
