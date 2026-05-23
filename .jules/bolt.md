## 2024-05-30 - Iterating pandas DataFrames efficiently
**Learning:** `pandas.DataFrame.iterrows()` is a major performance bottleneck for looping over datasets because it returns a Series for each row, creating significant overhead in Python.
**Action:** Always replace `iterrows()` with `itertuples(index=False, name=None)` for very fast, index-based tuple access, or `to_dict('records')` for dictionary key access, to eliminate DataFrame construction overhead during loops.
