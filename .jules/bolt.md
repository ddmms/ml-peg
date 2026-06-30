## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.

## 2024-06-30 - Faster Pandas Iteration
**Learning:** Iterating over Pandas DataFrames with `iterrows()` is a significant performance bottleneck (can take ~0.45s vs ~0.008s for `itertuples()` in some benchmarks). It yields `(index, Series)`.
**Action:** Always replace `iterrows()` with `itertuples(index=False, name=None)` for standard tuple iterations, or `to_dict('records')` when dictionary key lookups are needed, ensuring to shift indices appropriately (e.g. changing `row[1][0]` to `row[0]`).
