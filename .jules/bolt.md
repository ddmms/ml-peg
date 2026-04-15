## 2024-05-17 - Uncached YAML Parsing
**Learning:** Parsing `frameworks.yml` every time `get_framework_config` is called causes a performance bottleneck due to IO and YAML parsing on every function call.
**Action:** Use `@lru_cache(maxsize=1)` on `load_framework_registry` to cache the parsed YAML, significantly reducing the execution time. Need to `copy.deepcopy()` the results returned to prevent cache corruption, per project constraints.
