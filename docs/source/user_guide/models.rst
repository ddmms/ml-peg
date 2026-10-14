======
Models
======

The machine-learned interatomic potentials (MLIPs) known to ML-PEG are listed
below, with the configuration defined in ``ml_peg/models/models.yml``. Models
marked *(not currently enabled)* are commented out in ``models.yml`` and are not
run in the current benchmarks; they are listed here for completeness.

MACE
====

mace-mp-0a
----------

- `Docs <https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html#pretrained-mace-mp-models>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2025).
  `A foundation model for atomistic materials chemistry <https://doi.org/10.1063/5.0297006>`__.

- **Citation (architecture):** Batatia et al. (2022).
  `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields <https://openreview.net/forum?id=YPpSngE-ZU>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-mp %
      :end-before: % end mace-mp %

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace %
      :end-before: % end mace %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-mp-0a:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium"

mace-mp-0b3
-----------

- `Docs <https://github.com/ACEsuit/mace-foundations#latest-recommended-foundation-models>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2025).
  `A foundation model for atomistic materials chemistry <https://doi.org/10.1063/5.0297006>`__.

- **Citation (architecture):** Batatia et al. (2022).
  `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields <https://openreview.net/forum?id=YPpSngE-ZU>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-mp %
      :end-before: % end mace-mp %

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace %
      :end-before: % end mace %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-mp-0b3:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-0b3"

mace-mpa-0
----------

- `Docs <https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2025).
  `A foundation model for atomistic materials chemistry <https://doi.org/10.1063/5.0297006>`__.

- **Citation (architecture):** Batatia et al. (2022).
  `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields <https://openreview.net/forum?id=YPpSngE-ZU>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-mp %
      :end-before: % end mace-mp %

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace %
      :end-before: % end mace %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-mpa-0:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-mpa-0"

mace-omat-0
-----------

- `Docs <https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2025).
  `A foundation model for atomistic materials chemistry <https://doi.org/10.1063/5.0297006>`__.

- **Citation (architecture):** Batatia et al. (2022).
  `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields <https://openreview.net/forum?id=YPpSngE-ZU>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-mp %
      :end-before: % end mace-mp %

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace %
      :end-before: % end mace %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-omat-0:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "medium-omat-0"

mace-matpes-r2scan
------------------

- `Docs <https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2025).
  `A foundation model for atomistic materials chemistry <https://doi.org/10.1063/5.0297006>`__.
- **Citation (architecture):** Batatia et al. (2022).
  `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields <https://openreview.net/forum?id=YPpSngE-ZU>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-mp %
      :end-before: % end mace-mp %

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace %
      :end-before: % end mace %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-matpes-r2scan:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: r2SCAN
     kwargs:
       model: "mace-matpes-r2scan-0"
       head: "matpes_pbe"
     dispersion_kwargs:
       xc: r2scan
       label: D4

mace-mh-1-omat
--------------

- `Docs <https://huggingface.co/mace-foundations/mace-mh-1>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2025).
  `Cross Learning between Electronic Structure Theories for Unifying Molecular, Surface, and Inorganic Crystal Foundation Force Fields <https://arxiv.org/abs/2510.25380>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-mh %
      :end-before: % end mace-mh %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-mh-1-omat:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: "mh-1"
       head: omat_pbe


MACE-OFF23(L)
-------------

- `Docs <https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html#mace-off23-transferable-organic-force-fields>`__
- `Repo <https://github.com/ACEsuit/mace-off>`__
- **Citation:** Kovács et al. (2025).
  `MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules <https://doi.org/10.1021/jacs.4c07099>`__.

- **Citation (architecture):** Batatia et al. (2022).
  `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields <https://openreview.net/forum?id=YPpSngE-ZU>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-off %
      :end-before: % end mace-off %

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace %
      :end-before: % end mace %

**ML-PEG configuration:**

.. code-block:: yaml

   MACE-OFF23(L):
     module: mace.calculators
     class_name: mace_off
     device: "auto"
     trained_on_dispersion: true
     kwargs:
       model: large


MACE-OFF24(M)
-------------

- `Repo <https://github.com/ACEsuit/mace-off>`__
- **Citation:** Kovács et al. (2025).
  `MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules <https://doi.org/10.1021/jacs.4c07099>`__.

- **Citation (architecture):** Batatia et al. (2022).
  `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields <https://openreview.net/forum?id=YPpSngE-ZU>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-off %
      :end-before: % end mace-off %

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace %
      :end-before: % end mace %

**ML-PEG configuration:**

.. code-block:: yaml

   MACE-OFF24(M):
     module: mace.calculators
     class_name: mace_off
     device: "auto"
     trained_on_dispersion: true
     kwargs:
       model: "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model"


mace-omol
---------

- `Docs <https://github.com/ACEsuit/mace-foundations#latest-recommended-foundation-models>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2025).
  `Cross Learning between Electronic Structure Theories for Unifying Molecular, Surface, and Inorganic Crystal Foundation Force Fields <https://arxiv.org/abs/2510.25380>`__.
- **Citation (dataset):** Levine et al. (2025).
  `The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models <https://arxiv.org/abs/2505.08762>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-mh %
      :end-before: % end mace-mh %

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin omol25 %
      :end-before: % end omol25 %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-omol:
     module: mace.calculators
     class_name: mace_omol
     device: "auto"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model: "extra_large"

mace-mh-1-omol
--------------

- `Docs <https://huggingface.co/mace-foundations/mace-mh-1>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2025).
  `Cross Learning between Electronic Structure Theories for Unifying Molecular, Surface, and Inorganic Crystal Foundation Force Fields <https://arxiv.org/abs/2510.25380>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-mh %
      :end-before: % end mace-mh %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-mh-1-omol:
     module: mace.calculators
     class_name: mace_mp
     device: "auto"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model: "mh-1"
       head: omol

mace-polar-1-s
--------------

- `Docs <https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2026).
  `MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry <https://arxiv.org/abs/2602.19411>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-polar %
      :end-before: % end mace-polar %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-polar-1-s:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-s"

mace-polar-1-m
--------------

- `Docs <https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2026).
  `MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry <https://arxiv.org/abs/2602.19411>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-polar %
      :end-before: % end mace-polar %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-polar-1-m:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-m"

mace-polar-1-l
--------------

- `Docs <https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html>`__
- `Repo <https://github.com/ACEsuit/mace-foundations>`__
- **Citation:** Batatia et al. (2026).
  `MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry <https://arxiv.org/abs/2602.19411>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mace-polar %
      :end-before: % end mace-polar %

**ML-PEG configuration:**

.. code-block:: yaml

   mace-polar-1-l:
     module: mace.calculators
     class_name: mace_polar
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V
     kwargs:
       model: "polar-1-l"


DPA
===

Install these fixed-float32 DeePMD checkpoints with ``uv sync --extra dpa``.
DPA-3.3-1M uses OpenLAM-v1 with the OMat24 head and the DPA4 variants use OMat24.
DPA-3.3-1M is CC-BY-4.0, while DPA4 is CC-BY-NC-4.0 and restricted to
non-commercial use.

dpa-3p3-1M-omat
----------------

- `Docs <https://huggingface.co/deepmodelingcommunity/DPA-3.3-1M>`__
- `Repo <https://github.com/deepmodeling/deepmd-kit>`__
- **Citation:** Zhang et al. (2026).
  `A graph neural network for the era of large atomistic models <https://doi.org/10.1038/s41524-026-02146-2>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin dpa3 %
      :end-before: % end dpa3 %

**ML-PEG configuration:**

.. code-block:: yaml

   dpa-3p3-1M-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OpenLAM-v1, OMAT, MPtrj, OC20, OC22, ODAC23, SPICE2]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA-3.3-1M
       head: Omat24

dpa-4-nano-omat
---------------

- `Docs <https://huggingface.co/deepmodelingcommunity/DPA4-OMat24>`__
- `Repo <https://github.com/deepmodeling/deepmd-kit>`__
- **Citation:** Li et al. (2026).
  `DPA4: Pushing the Accuracy-Cost Frontier of Interatomic Potentials with EMFA SO(2) Convolution <https://arxiv.org/abs/2606.02419>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin dpa4 %
      :end-before: % end dpa4 %

**ML-PEG configuration:**

.. code-block:: yaml

   dpa-4-nano-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OMAT]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA4-Nano-OMat24-v20260805

dpa-4-neo-omat
--------------

- `Docs <https://huggingface.co/deepmodelingcommunity/DPA4-OMat24>`__
- `Repo <https://github.com/deepmodeling/deepmd-kit>`__
- **Citation:** Li et al. (2026).
  `DPA4: Pushing the Accuracy-Cost Frontier of Interatomic Potentials with EMFA SO(2) Convolution <https://arxiv.org/abs/2606.02419>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin dpa4 %
      :end-before: % end dpa4 %

**ML-PEG configuration:**

.. code-block:: yaml

   dpa-4-neo-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OMAT]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA4-Neo-OMat24-v20260805

dpa-4-plus-omat
---------------

- `Docs <https://huggingface.co/deepmodelingcommunity/DPA4-OMat24>`__
- `Repo <https://github.com/deepmodeling/deepmd-kit>`__
- **Citation:** Li et al. (2026).
  `DPA4: Pushing the Accuracy-Cost Frontier of Interatomic Potentials with EMFA SO(2) Convolution <https://arxiv.org/abs/2606.02419>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin dpa4 %
      :end-before: % end dpa4 %

**ML-PEG configuration:**

.. code-block:: yaml

   dpa-4-plus-omat:
     module: deepmd.calculator
     class_name: DP
     datasets: [OMAT]
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       model: DPA4-Plus-OMat24-v20260805

Orb
===

orb-v3-consv-inf-omat
---------------------

- `Docs <https://github.com/orbital-materials/orb-models>`__
- `Repo <https://github.com/orbital-materials/orb-models>`__
- **Citation:** Rhodes et al. (2025).
  `Orb-v3: atomistic simulation at scale <https://arxiv.org/abs/2504.06231>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin orb %
      :end-before: % end orb %

**ML-PEG configuration:**

.. code-block:: yaml

   orb-v3-consv-inf-omat:
     module: orb_models.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_conservative_inf_omat"

orb-v3-direct-inf-omat
----------------------

- `Docs <https://github.com/orbital-materials/orb-models>`__
- `Repo <https://github.com/orbital-materials/orb-models>`__
- **Citation:** Rhodes et al. (2025).
  `Orb-v3: atomistic simulation at scale <https://arxiv.org/abs/2504.06231>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin orb %
      :end-before: % end orb %

**ML-PEG configuration:**

.. code-block:: yaml

   orb-v3-direct-inf-omat:
     module: orb_models.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_direct_inf_omat"

orb-v3-consv-omol
-----------------

- `Docs <https://github.com/orbital-materials/orb-models>`__
- `Repo <https://github.com/orbital-materials/orb-models>`__
- **Citation (architecture):** Rhodes et al. (2025).
  `Orb-v3: atomistic simulation at scale <https://arxiv.org/abs/2504.06231>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin orb %
      :end-before: % end orb %

**ML-PEG configuration:**

.. code-block:: yaml

   orb-v3-consv-omol:
     module: orb_models.forcefield.inference.calculator
     class_name: OrbCalc
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: PBE
     kwargs:
       name: "orb_v3_conservative_omol"


PET
===

pet-mad
-------

- `Docs <https://lab-cosmo.github.io/upet/latest/models.html>`__
- `Repo <https://github.com/lab-cosmo/upet>`__
- **Citation:** Mazitov et al. (2025).
  `PET-MAD as a lightweight universal interatomic potential for advanced materials modeling <https://doi.org/10.1038/s41467-025-65662-7>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin pet-mad %
      :end-before: % end pet-mad %

**ML-PEG configuration:**

.. code-block:: yaml

  pet-mad:
    module: upet.calculator
    class_name: UPETCalculator
    device: "cpu"
    trained_on_dispersion: false
    level_of_theory: PBEsol
    kwargs:
      model: "pet-mad-s"
      version: "1.0.2"
    dispersion_kwargs:
      xc: pbesol

pet-mad-s-1p5
-------------

- `Docs <https://lab-cosmo.github.io/upet/latest/models.html>`__
- `Repo <https://github.com/lab-cosmo/upet>`__
- **Citation:** Malosso et al. (2026).
  `High-quality, high-information datasets for universal atomistic machine learning <https://arxiv.org/abs/2603.02089>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin pet-mad-1p5 %
      :end-before: % end pet-mad-1p5 %

**ML-PEG configuration:**

.. code-block:: yaml

  pet-mad-s-1p5:
    module: upet.calculator
    class_name: UPETCalculator
    device: "cpu"
    trained_on_dispersion: false
    level_of_theory: r2SCAN
    kwargs:
      model: "pet-mad-s"
      version: "1.5.0"
    dispersion_kwargs:
      xc: r2scan

pet-oam-xl
----------

- `Docs <https://lab-cosmo.github.io/upet/latest/models.html>`__
- `Repo <https://github.com/lab-cosmo/upet>`__
- **Citation:** Bigi et al. (2026).
  `Pushing the limits of unconstrained machine-learned interatomic potentials <https://doi.org/10.1088/2632-2153/ae6417>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin pet-oam %
      :end-before: % end pet-oam %

**ML-PEG configuration:**

.. code-block:: yaml

  pet-oam-xl:
    module: upet.calculator
    class_name: UPETCalculator
    device: "cpu"
    trained_on_dispersion: False
    level_of_theory: PBE
    kwargs:
      model: "pet-oam-xl"
      version: "1.0.0"


UMA (FairChem)
==============

uma-s-1p1-omat
--------------

- `Docs <https://huggingface.co/facebook/UMA>`__
- `Repo <https://github.com/facebookresearch/fairchem>`__
- **Citation:** Wood et al. (2025).
  `UMA: A Family of Universal Models for Atoms <https://papers.nips.cc/paper_files/paper/2025/hash/bbf23e81b0ad7637fe9a731d0b676ca6-Abstract-Conference.html>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin uma %
      :end-before: % end uma %

**ML-PEG configuration:**

.. code-block:: yaml

   uma-s-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     level_of_theory: PBE
     trained_on_dispersion: false
     kwargs:
       model_name: "uma-s-1p1"
       task_name: "omat"

uma-m-1p1-omat
--------------

- `Docs <https://huggingface.co/facebook/UMA>`__
- `Repo <https://github.com/facebookresearch/fairchem>`__
- **Citation:** Wood et al. (2025).
  `UMA: A Family of Universal Models for Atoms <https://papers.nips.cc/paper_files/paper/2025/hash/bbf23e81b0ad7637fe9a731d0b676ca6-Abstract-Conference.html>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin uma %
      :end-before: % end uma %

**ML-PEG configuration:**

.. code-block:: yaml

   uma-m-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     level_of_theory: PBE
     trained_on_dispersion: false
     kwargs:
       model_name: "uma-m-1p1"
       task_name: "omat"

uma-s-1p1-omol
--------------

- `Docs <https://huggingface.co/facebook/UMA>`__
- `Repo <https://github.com/facebookresearch/fairchem>`__
- **Citation:** Wood et al. (2025).
  `UMA: A Family of Universal Models for Atoms <https://papers.nips.cc/paper_files/paper/2025/hash/bbf23e81b0ad7637fe9a731d0b676ca6-Abstract-Conference.html>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin uma %
      :end-before: % end uma %

**ML-PEG configuration:**

.. code-block:: yaml

   uma-s-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-s-1p1"
       task_name: "omol"

uma-s-1p2-omol
--------------

- `Docs <https://huggingface.co/facebook/UMA>`__
- `Repo <https://github.com/facebookresearch/fairchem>`__
- **Citation:** Wood et al. (2025).
  `UMA: A Family of Universal Models for Atoms <https://papers.nips.cc/paper_files/paper/2025/hash/bbf23e81b0ad7637fe9a731d0b676ca6-Abstract-Conference.html>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin uma %
      :end-before: % end uma %

**ML-PEG configuration:**

.. code-block:: yaml

   uma-s-1p2-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-s-1p2"
       task_name: "omol"

uma-m-1p1-omol
--------------

- `Docs <https://huggingface.co/facebook/UMA>`__
- `Repo <https://github.com/facebookresearch/fairchem>`__
- **Citation:** Wood et al. (2025).
  `UMA: A Family of Universal Models for Atoms <https://papers.nips.cc/paper_files/paper/2025/hash/bbf23e81b0ad7637fe9a731d0b676ca6-Abstract-Conference.html>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin uma %
      :end-before: % end uma %

**ML-PEG configuration:**

.. code-block:: yaml

   uma-m-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-m-1p1"
       task_name: "omol"


MatterSim
=========

mattersim-5M
------------

- `Docs <https://microsoft.github.io/mattersim/>`__
- `Repo <https://github.com/microsoft/mattersim>`__
- **Citation:** Yang et al. (2024).
  `MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures <https://arxiv.org/abs/2405.04967>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mattersim %
      :end-before: % end mattersim %

**ML-PEG configuration:**

.. code-block:: yaml

   mattersim-5M:
     module: mattersim.forcefield
     class_name: MatterSimCalculator
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       load_path: "mattersim-v1.0.0-5m"

mattersim-1M
------------

- `Docs <https://microsoft.github.io/mattersim/>`__
- `Repo <https://github.com/microsoft/mattersim>`__
- **Citation:** Yang et al. (2024).
  `MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures <https://arxiv.org/abs/2405.04967>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin mattersim %
      :end-before: % end mattersim %

**ML-PEG configuration:**

.. code-block:: yaml

   mattersim-1M:
     module: mattersim.forcefield
     class_name: MatterSimCalculator
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       load_path: "mattersim-v1.0.0-1m"


GRACE
=====

GRACE-2L-OMAT
-------------

- `Docs <https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/>`__
- `Repo <https://github.com/ICAMS/grace-tensorpotential>`__
- **Citation:** Lysogorskiy et al. (2026).
  `Graph atomic cluster expansion for foundational machine learning interatomic potentials <https://doi.org/10.1038/s41524-026-01979-1>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin grace %
      :end-before: % end grace %

**ML-PEG configuration:**

.. code-block:: yaml

  grace-2l-omat:
    module: tensorpotential.calculator
    class_name: grace_fm
    device: "cpu"
    trained_on_dispersion: false
    kwargs:
      model: GRACE-2L-OMAT


GRACE-2L-SMAX-OMAT-medium
-------------------------

- `Docs <https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/>`__
- `Repo <https://github.com/ICAMS/grace-tensorpotential>`__
- **Citation:** Bochkarev et al. (2026).
  `Exploring the extremes: atomic basis for multi-elemental materials science under complex thermodynamic conditions <https://arxiv.org/abs/2602.23489>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin grace-smax %
      :end-before: % end grace-smax %

**ML-PEG configuration:**

.. code-block:: yaml

  grace-2l-smax-omat-medium:
    module: tensorpotential.calculator
    class_name: grace_fm
    device: "cpu"
    trained_on_dispersion: false
    kwargs:
      model: GRACE-2L-SMAX-OMAT-medium


SevenNet
========

sevennet-omni-mpa
-----------------

- `Docs <https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html#sevennet-omni>`__
- `Repo <https://github.com/MDIL-SNU/SevenNet>`__
- **Citation:** Kim et al. (2026).
  `Optimizing cross-domain transfer for universal machine learning interatomic potentials <https://doi.org/10.1038/s41467-026-70195-8>`__.

.. details:: BibTeX

   .. literalinclude:: model_references.bib
      :language: bibtex
      :start-after: % begin sevennet %
      :end-before: % end sevennet %

**ML-PEG configuration:**

.. code-block:: yaml

  sevennet-omni-mpa:
    module: sevenn.sevennet_calculator
    class_name: SevenNetCalculator
    device: "cpu"
    trained_on_dispersion: false
    level_of_theory: PBE
    kwargs:
      model: "7net-omni"
      modal: "mpa"
