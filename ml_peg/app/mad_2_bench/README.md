# MAD-2 test

Errors of models against six DFT references on structures from the MAD-2 test set.
The inputs, one extended xyz file per reference and `protocols.json`, are downloaded
from `inputs/mad_2_bench/errors/mad-2-bench.zip` in the ML-PEG S3 bucket. Energies are
compared only where `ml_peg/analysis/mad_2_bench/errors/model_metadata.yml` says the
training energies of a model follow the same DFT setup as the label.

```bash
ml_peg calc --category mad_2_bench --test errors --models <model>
ml_peg analyse --category mad_2_bench --test errors --models <models>
ml_peg app --category mad_2_bench
```

The calculation writes `ml_peg/calcs/mad_2_bench/errors/outputs/<model>/predictions.jsonl`.
The analysis writes the data of both pages to `ml_peg/app/data/mad_2_bench`.
