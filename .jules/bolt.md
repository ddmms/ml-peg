## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.

## 2024-05-19 - Replacing Pandas iterrows with itertuples/to_dict
**Learning:** Pandas `iterrows()` is a known performance anti-pattern that creates significant overhead by yielding pandas Series objects. Replacing it with `itertuples(index=False)` (yielding namedtuples) or `to_dict('records')` (when column names are dynamic or invalid identifiers) speeds up iteration over dataframes by >30x.
**Action:** Never use `iterrows()` for dataframe iteration. Use `itertuples(index=False)` for static known columns, or `to_dict('records')` if keys are dynamic. Adjust row index accesses accordingly (e.g. `row[1][0]` becomes `row[0]`).
