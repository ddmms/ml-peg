=================
Adding categories
=================

Categories are automatically inferred from subdirectories in ``ml_peg/calcs``,
``ml_peg/analysis``, and ``ml_peg/app``, as long as these subdirectories contain
at least one additional subdirectory with the files relevenat files, as described in
:doc:`Adding benchmarks </developer_guide/add_benchmarks>`.

For example, to add the ``new_category`` category in ``ml_peg/calcs``, first create the
``ml_peg/calcs/new_category`` directory, then create the directory for a new test
and its calculation script, ``ml_peg/calcs/new_category/new_test/calc_new_test.py``.

For calculations and analaysis, these categories are largely for organisation of code.
However, in the case of the test apps, categories also define the tabs and
summarisation of data, as described in :doc:`About ML-PEG </user_guide/about>`.

Additionally, a yaml file can be provided for the category, providing a human-friendly
name, used for the tab name and title, as well as a description of the category. For
example::

    title: New category
    description: New category description

This file should be placed in ``ml_peg/app/[category]``, and named ``[category].yml``.

If this file is missing, or is not read successfully, the category name will default to
the direcory name, and the description will be left empty.

This file also sets the category's default ``weight`` in the overall score, and the
default ``benchmark_weights`` of the benchmarks within it. Note that benchmarks marked
slow or slower must be weighted 0 here. See
:doc:`Benchmark scoring </developer_guide/scoring_and_normalisation>`.
