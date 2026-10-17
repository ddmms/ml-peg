# _mock

Mock container for testing the optimised inference speed pipeline without a GPU.
It implements contract v1.0 with the Python standard library only: it does no MD,
sleeps briefly, and writes a valid `inference.json` and, for `validate: true`
sizes, a `validation.extxyz` with zero energy and forces. Its reference is ML-PEG's
`mock` model, which also returns zeros.

Failures are planted with `MLPEG_MOCK_FAIL`, a `;`-separated list of
`<fault>:<run-id glob>` entries; the faults are listed in `runscript.py`. Pass it
with `--container-env`, for example:

```bash
ml_peg inference-speed run --campaign campaign --models _mock --runtime local \
  --container-env "MLPEG_MOCK_FAIL=oom:*-n6144-*;drift:*-r2"
```

`--runtime local` runs `runscript.py` directly. To use Apptainer instead:

```bash
ml_peg inference-speed build --model _mock
ml_peg inference-speed run --campaign campaign --models _mock
```
