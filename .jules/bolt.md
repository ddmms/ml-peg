## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-20 - Efficient Caching for Config Access
**Learning:** Returning a `deepcopy` of an entire large cached registry dictionary on every call just to access a single entry is highly inefficient.
**Action:** When caching functions that return large configurations, do not use `deepcopy` on the entire result. Instead, cache the getter function directly (`get_framework_config`), or pull the specific entry from the un-copied cached registry and only deepcopy that specific entry. Use `@functools.cache` instead of `@lru_cache(maxsize=None)` as enforced by Ruff rule UP033.
