## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.

## 2024-05-20 - Fast DataFrame Iteration Avoids Overhead
**Learning:** Iterating over Pandas DataFrames using `.iterrows()` introduces significant overhead because it creates a new `pd.Series` object for every single row. In calculation scripts (like elasticity, solvMPCONF196, etc.), this causes notable slowdowns when processing large datasets or many files.
**Action:** Replace `.iterrows()` with `.itertuples(index=False)` (or `name=None` if accessing by index `row[0]`) for purely positional or namedtuple-based attribute access, which is drastically faster. If variable column names require string keys, use `for row in df.to_dict('records'):`. Always check if `.iterrows()` is a bottleneck and optimize it.
