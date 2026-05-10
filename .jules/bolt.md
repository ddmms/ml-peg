## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.

## 2024-05-19 - DataFrame Iteration Bottleneck
**Learning:** Using `iterrows()` on Pandas DataFrames is a severe performance bottleneck because it boxes each row into a Series object, adding massive overhead. Testing showed `iterrows()` took ~84ms for 1000 rows compared to `itertuples()` at ~2ms and `dict(zip(...))` at ~0.8ms (a ~100x speedup).
**Action:** Never use `iterrows()`. When iterating over DataFrames, use `itertuples(index=False)` and access via namedtuple attributes, or convert to dictionary directly via `dict(zip(df.iloc[:, 0], df.iloc[:, 1]))` if mapping two columns. If dynamic column names are needed from tuples, use `row._asdict()`.
