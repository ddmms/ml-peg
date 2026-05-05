## 2024-05-19 - Caching YAML Load for Model Parsing
**Learning:** `yaml.safe_load` on large configuration files like `models.yml` is significantly slower than parsing JSON or doing other basic IO. It can become a bottleneck when called repeatedly throughout an application's lifecycle (e.g., getting subsets of models, instantiating apps).
**Action:** Always memoize or `@lru_cache` functions that load static, read-only configuration files (like `models.yml`) to prevent repeated disk I/O and parsing overhead.

## 2024-05-20 - Caching YAML Load for FAQs and Frameworks
**Learning:** Functions that parse static configuration files using `yaml.safe_load` repeatedly can be unexpectedly slow and act as bottlenecks. Like `models.yml`, `frameworks.yml` and `faqs.yml` also suffer from this.
**Action:** When working on apps that use utility functions to parse YAML configs (e.g. badges or UI configuration), ensure the read/parse operations are cached using `@functools.cache`.
