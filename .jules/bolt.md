## 2025-02-12 - Avoid np.isnan with lists for scalar checks
**Learning:** Checking for NaNs in scalars using `np.isnan([a, b, c]).any()` incurs significant overhead from list allocation and array conversion, taking ~20x longer than using `math.isnan(a) or math.isnan(b) or math.isnan(c)`.
**Action:** For simple scalar validation where types are handled or fallbacks exist, prefer `math.isnan` coupled with `try/except TypeError` over constructing NumPy arrays.
