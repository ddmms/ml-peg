## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2025-01-26 - Fast info loading from Extended XYZ files
**Learning:** In Extended XYZ files (EXYZ) used by the `ase` library, the `atoms.info` metadata is stored on the second line. Reading these properties using `ase.io.read()` has a large overhead since it reads the whole structure.
**Action:** When unit testing or retrieving simple metadata dynamically from a batch of `.xyz` files (e.g. loops in `get_system_names()`), avoid `ase.io.read()` and use the fast extraction utility `read_extxyz_info_fast` using `ase.io.extxyz.key_val_str_to_dict`.
