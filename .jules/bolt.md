## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.
## 2024-05-19 - Caching YAML Load for FAQs
**Learning:** `yaml.safe_load` on `faqs.yml` within `build_faqs()` was taking ~0.7 seconds per call on application build, causing a major UI slowdown.
**Action:** Applied the `@functools.cache` and `deepcopy` pattern successfully to an internal `_load_faqs_registry()` function to avoid caching a mutable list directly, reducing execution to ~0.07 seconds. Also patched `__init__.py` to gracefully handle metadata lookup failures when the package is tested locally outside an installation.
