## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.

## 2024-05-19 - Caching ase.Atoms and Clearing Metadata for I/O
**Learning:** Re-reading the same XYZ files within nested conformer loops (like in MPCONF196 benchmarks) is a significant disk I/O bottleneck. Caching `ase.Atoms` objects in a list during the first pass avoids this. However, to preserve translations and correctly output just the structure (without stale energy/forces metadata), you MUST explicitly clear the calculator (`atoms.calc = None`) before writing to disk using `ase.io.write`.
**Action:** When reusing `ase.Atoms` objects across calculation loops, cache them to avoid duplicate disk reads, but always set `atoms.calc = None` prior to final structure serialization if clean output is required.
