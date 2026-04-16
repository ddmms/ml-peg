## 2024-04-16 - Cache YAML Reads
**Learning:** Loading and parsing YAML configuration files in Python (like `models.yml`) without caching creates a measurable bottleneck during frequent invocations in this codebase (e.g., repeating loads in analysis or app building loops), causing excessive I/O overhead.
**Action:** Use `functools.lru_cache` to cache parsed YAML dictionaries. Always use `copy.deepcopy()` when returning these cached mutable objects to prevent unintended state mutation by callers.
