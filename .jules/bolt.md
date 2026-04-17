## 2026-04-17 - lru_cache with mutable objects
**Learning:** Using `@lru_cache` on functions returning mutable objects like parsed YAML dicts can lead to cache corruption if callers modify the returned dictionary.
**Action:** When using `@lru_cache` on functions returning mutable objects, ensure that `copy.deepcopy()` is applied on the returned value by callers or inside the function before returning.
