=================
Benchmark scoring and normalisation
=================


This guide will break down how benchmark metrics are scored and normalised by default and how users can specify their own normalisation schemes.
Normalisation of benchmark metrics to yield a unitless score is essential if we want to combine multiple metrics of different units and order of magnitude into a single overall benchmark score.


- link to developer_guide for custom normalisation schemes


Default scoring and normalisation
-------------------

By default every benchmark metric is normalised to a unitless score between 0 (worst) and 1 (best) using "good" and "bad" thresholds specified in the metric configuration file (metrics.yml).
The "good" threshold defines a raw metric value, where performance better than this value is considered as good as a user would require.
The "bad" threshold defines a raw metric value, where performance worse than this value indicates that the model should not be used for the intended application.


The normalisation is linear between these thresholds, with scores capped at 0 and 1 outside the thresholds. For example, if a metric has a "good" threshold of 1.0 and a "bad" threshold of 5.0, a raw metric value of 1.0 or lower will yield a score of 1.0, a raw value of 5.0 or higher will yield a score of 0.0, and a raw value of 3.0 will yield a score of 0.5.

These thresholds have been carefully chosen for each metric to help users decide in which situtations foudation models perform well and are reliable to use, and in which situations they perform poorly and should be treated with caution or avoided completely. The ability to tune these thresholds also allows experts in specific fields to adapt the scoring to their own requirements and expectations.
