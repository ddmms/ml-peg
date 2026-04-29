## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.
## 2024-05-19 - Caching YAML Load for Framework Parsing
**Learning:** Similar to `models.yml`, the application frequently loads `frameworks.yml` via `load_framework_registry` on page loads and during app building. Parsing YAML files is a known bottleneck.
**Action:** When working with static, read-only configuration YAML files, ensure they are wrapped with an `@lru_cache` and that any public interface returns a `copy.deepcopy()` to prevent unintended mutations from rippling back into the cached object.
