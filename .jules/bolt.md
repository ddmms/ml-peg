## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.
## 2024-05-20 - Fast EXYZ Metadata Parsing
**Learning:** `ase.io.read` on large extended XYZ files is very slow if you only need the `atoms.info` metadata because it parses all atomic coordinates and structure data. For analysis scripts that only iterate to extract energies or metrics, this can become a huge bottleneck.
**Action:** When extracting only the metadata, use a fast custom parser (like `ase.io.extxyz.key_val_str_to_dict` applied directly to the second line of the file).
