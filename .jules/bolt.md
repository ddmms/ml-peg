## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.
## 2024-05-19 - [Replace iterrows with faster iteration]
**Learning:** Pandas `iterrows()` is extremely slow for looping over rows. Using `itertuples()` for structured properties and positional access or `to_dict('records')` when column names are dynamic or non-standard identifiers yields a 10-50x speedup with minimal effort.
**Action:** Always prefer `itertuples()` or `to_dict('records')` over `iterrows()`. If index-based loop variables are needed (e.g. `row[0]`), use `itertuples(index=False, name=None)`. If column names are guaranteed to be valid Python identifiers, use `itertuples(index=False)` and access via `row.ColName`. If column names are dynamic or invalid identifiers, use `to_dict('records')`.
