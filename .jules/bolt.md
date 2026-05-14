## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-19 - Caching YAML Load for Framework Registry
**Learning:** `yaml.safe_load` on `frameworks.yml` within `load_framework_registry()` was taking ~2-3 ms per call and it was repeatedly called for every framework entry via `get_framework_config()`. This was a micro-bottleneck, especially when dealing with lists or multiple frameworks.
**Action:** Applied the `@lru_cache` and `deepcopy` pattern successfully again to `load_framework_registry()` and `get_framework_config()` to avoid caching a mutable dictionary directly and avoid repeated YAML I/O parsing.

## 2024-05-19 - Pandas Iteration Bottlenecks
**Learning:** `iterrows()` is consistently used across the codebase (e.g., `calc_solvMPCONF196.py`, `gscdb138.py`) for iterating through DataFrames and is a major, known performance bottleneck (often 10-20x slower than alternatives).
**Action:** Replace `iterrows()` with `itertuples(index=False, name=None)` when simple tuple indexing is sufficient, standard `itertuples()` for dot-notation access, or `to_dict('records')` when dictionary access patterns like `.get()` are required by downstream logic.

## 2024-05-19 - Caching UI Layout Generation
**Learning:** The `build_faqs()` component function was reading `faqs.yml` from disk synchronously on every render without caching, similar to previous issues discovered with `frameworks.yml`.
**Action:** Apply `@functools.cache` to UI component generation functions that depend on static configuration files to eliminate repetitive disk I/O and parsing overhead.

## 2024-05-19 - Hanging Tests in Restricted Network
**Learning:** Running Pytest on heavy ML integration tests (like those in `calc_solvMPCONF196.py` and `calc_high_pressure_relaxation.py`) in environments with restricted network access causes the test suite to hang or timeout as the system attempts to download gigabytes of model weights (e.g., Torch, Mace) silently in the background.
**Action:** When tests hang in this manner due to missing heavy dependencies that cannot be easily installed, use static analysis (`ruff check`) and `python -m py_compile` as the primary verification strategy to ensure the syntax and logic of refactored code is sound without triggering remote downloads.
