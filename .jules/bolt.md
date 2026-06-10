## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.

## 2026-06-10 - Replace iterrows with itertuples or to_dict('records')
**Learning:** pandas `iterrows()` is a significant performance bottleneck. Tests revealed `to_dict('records')` and `itertuples()` are orders of magnitude faster when iterating large DataFrames, reducing iteration times from seconds down to milliseconds.
**Action:** Use `df.itertuples(index=False, name=None)` for indexed access or `df.to_dict('records')` for dictionary access with dynamic column keys instead of `iterrows()`.
