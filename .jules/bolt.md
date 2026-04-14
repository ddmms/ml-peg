## 2024-05-18 - Caching YAML loading
**Learning:** Repeated YAML loading (`yaml.safe_load`) on each function call to load static configuration files (like `frameworks.yml`) can slow down applications significantly. Because `frameworks.yml` doesn't change during app runtime, we can safely cache its contents.
**Action:** Apply `@lru_cache(maxsize=1)` to configuration loading functions like `load_framework_registry` to cache the file contents, preventing unnecessary disk I/O and parsing time.
## 2024-05-18 - Caching mutable objects
**Learning:** Returning mutable objects (like a `dict`) from an `@lru_cache`-decorated function causes the caller to modify the *cached instance itself* if it mutates the dictionary, corrupting the cache for all future callers.
**Action:** Always return a deep copy (e.g. `copy.deepcopy()`) of the cached mutable object or use immutable structures, ensuring subsequent requests get clean, uncorrupted data.
