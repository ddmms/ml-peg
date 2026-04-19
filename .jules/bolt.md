## 2025-04-19 - Caching YAML Parsing

**Learning:** `yaml.safe_load` is extremely slow compared to `copy.deepcopy()`. When multiple functions in the codebase read the same static configuration file (`models.yml`), parsing it repeatedly becomes a noticeable bottleneck, taking ~10ms per load vs <0.1ms for a deepcopy of a cached dictionary.
**Action:** Use `@lru_cache` for functions that load static YAML files, but ensure callers use `copy.deepcopy()` on the cached output to prevent unintended mutation of the cache by downstream code.
