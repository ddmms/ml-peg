# Polymer Sets

These files define the polymer subsets.

- `small.txt`: 7 polymers common to all precomputed CSV outputs.
- `medium.txt`: 30 polymers common to `MACE-OFF23-S`, `PCFF`, and `vivance`.
- `large.txt`: 85 polymers common to `MACE-OFF23-S`, `PCFF`, and `vivance`.
- `x-large.txt`: the full 130-polymer set

Each file contains one polymer ID per line. Use the line number as the array
task index, for example:

```bash
POLY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" medium.txt)
```
