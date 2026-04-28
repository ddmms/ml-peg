## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.
## 2024-05-19 - Efficient Extxyz Metadata Parsing
**Learning:** `ase.io.read` parses the full atomic structure (all coordinates) which is extremely slow when only the file's metadata (`atoms.info`) is required.
**Action:** When only metadata is needed from `.extxyz` files, use `ml_peg.analysis.utils.utils.read_extxyz_info_fast` which reads the first two lines and parses the `info` string directly using `ase.io.extxyz.key_val_str_to_dict`.
