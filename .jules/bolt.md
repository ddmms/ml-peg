## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-20 - Cache Mutation on Mutable Return Values
**Learning:** Returning a cached dictionary from a configuration file parser can lead to subtle cross-contamination bugs if downstream users mutate the retrieved dictionary. Since `functools.cache` stores a reference to the mutable object, changes made to it will reflect in subsequent calls to the cached function.
**Action:** Always ensure that when caching functions that return mutable structures like parsed YAML dictionaries, functions retrieving these configs deep-copy the specific entries accessed, using `copy.deepcopy()`, to preserve the pristine state of the cache.
