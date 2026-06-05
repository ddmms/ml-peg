## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.

## 2024-06-05 - Avoid Redundant Structure Parsing in Benchmark Loops
**Learning:** Benchmark loops that calculate conformer energies and subsequently write annotated structures to disk often repeat expensive `get_atoms()` (file I/O) and translation operations in consecutive loops over the same set of molecules. In addition, computing array means (e.g., `np.mean`) inside the final loop causes redundant computation.
**Action:** Cache the parsed `ase.Atoms` objects in a list during the first pass and iterate over them in the second pass using `zip`. Hoist constant mean calculations (like `np.mean(abs_energies)`) outside of loops. Before writing cached atoms, explicitly clear their calculator (`atoms.calc = None`) to write structural changes without evaluating stale calculators.
