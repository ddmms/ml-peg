## 2025-04-13 - Cached model configuration loading
**Learning:** The application calls `get_model_names` frequently during startup (it's called in many files at import time). This repeatedly parsed the `models.yml` YAML file, significantly slowing down the app load time due to YAML parsing overhead.
**Action:** Used `functools.lru_cache` to cache the loaded dictionary from `models.yml` and reuse it across `get_model_names` calls.
