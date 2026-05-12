# Polymer Sets

These files define the polymer subsets.

- `small.txt`: 10 polymers. This is a hand-picked subset:
  `PE`, `PIB`, `PET`, `PCTFE`, `PS`, `PVMS`, `PVC`, `PAN`, `PTPC`, `PODPM`.
- `medium.txt`: 50 polymers. This is the `small.txt` subset, followed by the
  first 40 polymers from `../data.csv` that are not already in `small.txt`.
- `large.txt`: 130 polymers. This contains all polymer IDs from `../data.csv`.

Each file contains one polymer ID per line. The line number is the one-based
`--poly-index` used by `ml_peg calc`, so Slurm array jobs can pass the array
task id directly:

```bash
ml_peg calc \
  --category molecular_dynamics \
  --test polymers \
  --models mace-mp-0a \
  --poly-set medium \
  --poly-index "${SLURM_ARRAY_TASK_ID}"
```
