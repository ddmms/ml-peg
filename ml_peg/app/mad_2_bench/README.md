# MAD-2 test

Errors of models against six DFT references on structures from the MAD-2 test set.
The inputs, one extended xyz file per reference and `protocols.json`, are downloaded
from `inputs/mad_2_bench/errors/errors.zip` in the ML-PEG S3 bucket.

```bash
ml_peg calc --category mad_2_bench --test errors --models <model>
ml_peg analyse --category mad_2_bench --test errors --models <models>
ml_peg app --category mad_2_bench
```

The calculation writes `ml_peg/calcs/mad_2_bench/errors/outputs/<model>/predictions.jsonl`.
The analysis writes the data of both pages to `ml_peg/app/data/mad_2_bench`.

## Model metadata

Forces and stress are compared for every model. Energies are compared only where the
training energies of a model follow the same DFT setup as the label, because energies
from different POTCARs, Hubbard U values or energy definitions differ by a
composition-dependent shift that is not a model error.

`ml_peg/analysis/mad_2_bench/errors/model_metadata.yml` holds what the analysis needs
to decide this. It has one record per model, under the model's name in
`ml_peg/models/models.yml`.

| Field | Meaning |
| --- | --- |
| `name`, `checkpoint`, `training_data` | Text of the model tooltip. |
| `energy_relation` | How the training energies relate to each reference. Energies are compared for `same-family` and `u-differs` only. |
| `energy_scale_sources` | References whose DFT setup the training energies follow. Each frame is compared only if POTCARs, Hubbard U and energy definition of that setup match the label. |
| `energy_corrections` | Corrections applied to the training energies. Energies are not compared while this starts with `unknown`. |
| `mad_energy_target` | MAD-2 label the energy is compared with, `energy` or `atomization_energy`. |
| `molecular_only` | Model trained on isolated molecules only. Hidden on periodic structures unless out-of-domain models are included. |
| `supported_elements` | Elements the model covers, `null` for all. Other frames count as unsupported, not as failures. |
| `stress` | Whether the stress predictions of the model are scored. |

A model without a record is still scored on forces and stress. Its energies are not
compared and its tooltip says "not documented". To add a model, add a record under its
registry name.
