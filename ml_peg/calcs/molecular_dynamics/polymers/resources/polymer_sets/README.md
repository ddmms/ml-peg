# Polymer Sets

These files define the polymer subsets.

- `small.txt`: 10 polymers. This is a hand-picked subset:
  `PE`, `PIB`, `PET`, `PCTFE`, `PS`, `PVMS`, `PVC`, `PAN`, `PTPC`, `PODPM`.
- `medium.txt`: 50 polymers. This is the `small.txt` subset, followed by the
  first 40 polymers from `../data.csv` that are not already in `small.txt`.
- `large.txt`: 130 polymers. This contains all polymer IDs from `../data.csv`.

Each file contains one polymer ID per line. Use the line number as the array
task index, for example:

```bash
POLY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" medium.txt)
```
