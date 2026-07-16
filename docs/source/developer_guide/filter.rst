=================
Element filtering
=================

ML-PEG provides the ability to filter each benchmark by elements. In its most basic
form, this works by evaluating whether selected elements are required to carry out the
entirity of a benchmark, and if they are, the benchmark is excluded from calculated
scores.

In future, partial filtering will be supported, where individual data points can be
excluded from calculation scores.

In either case, several checks and modifications should be made when adding a new
benchmark, as described below.


Calculations
------------

The key principle behind changes to calculations is to provide a more consistent output
between models. Calculations should typically be designed to catch, rather than raise
errors. For example:

.. code-block:: python

    for atoms in atoms_list:
        try:
            atoms.get_potential_energy()
        except Exception as exc:
            warn(f"Error calculating energy: {exc}", stacklevel=2)
            atoms.info["energy"] = np.nan


Although this can also be handled during the analysis, catching errors in this way
makes it simpler to compare outputs between models. For tests that investigate multiple
structures and/or properties, this also ensures the calculation is not interupted,
which will be important for future partial filtering.

This can be tested using the ``MockErrorCalculator``:

.. code-block:: bash

    ml_peg calc --test [test] --models-file ml_peg/models/mock_error.yml --no-run-mock


which will typically store results in
``ml_peg/calcs/[category]/[test]/outputs/mock-error``.

In order to provide a baseline, the ``MockCalculator`` can be used to return 0 for all
energies, forces, and stresses. This can be run similarly to above:

.. code-block:: bash

    ml_peg calc --test [test] --models-file ml_peg/models/mock.yml


However, this calculator is also run by default when running a calculation, and can be
run without any other models by using the ```--mock-only`` flag:

.. code-block:: bash

    ml_peg calc --test [test] --mock-only


Running the ``MockCalculator`` can be skipped ``--no-run-mock`` when calling
``ml_peg calc``, as we shown above for the ``MockErrorCalculator``.

In most cases, the results produced by ``MockCalculator`` will be to save elemental
information that the application needs for filtering, as well additional information,
such as labels for hoverdata.


Analysis
--------

Analyses must be updated to ensure they save the elemental information required for
filtering. This typically involves using one of two utility functions. If only a single
list of elements needs to be extracted from the structure data, ``write_struct_info``
can be used:

.. code-block:: python

    from ml_peg.app.utils.utils import write_struct_info

    def test_benchmark(metrics):
        write_struct_info(
            data_path=CALC_PATH / "mock" / "struct.extxyz",
            out_path=OUT_PATH,
            index=0,
        )


This function takes a structure file, or list of structure files, and saves the
combined set of elements found to ``OUT_PATH / "info.json"``.

While this all that is strictly required for basic filtering, in preparation for
partial filtering, it is encouraged to save finer-grained elemental information where
relevant. For example, if a benchmark iterates through many structures, a list of
elements should be saved for each. This can be done using ``get_struct_info``:

.. code-block:: python

    from ml_peg.app.utils.utils import get_struct_info

    SYSTEM_INFO = get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="*.xyz",
        index="0",
        include_filenames=True,
        out_path=OUT_PATH,
        info_keys=["sys_formula"],
    )


Since tests of this form often require labels to be extracted for hoverdata, this
function also provides the option to save lists of keys saved to ``Atoms.info`` through
 the ``info_keys`` option, as well as the filename stems using ``include_filenames``.

Instead of a single list of elements, this will save elements as a list of lists in
``OUT_PATH / "info.json"``, as well as storing the other extracted informaiton in the
same file.

The main other change that may be required is to ensure analyses can handle missing files
and invalid values when calculating metrics. In these cases, metric values should be set to
NaN/None (in either case, they will be treated as a NaN when calculating scores).


Application
-----------

Changes to the application focus on loading the elemental information produced during
analysis, and defining functions to use this information to filter the metrics.

Currently, relatively minimal changes are needed to enable filtering for each
application, as this is largely handled behind-the-scenes. When instaniating a test app,
``info_path`` should be set to the path of the ``info.json`` that was produced during
analysis (typically, ``DATA_PATH / "info.json"``).

``BaseApp``, which each test app should inherit from, defines functions to load this
file. It then uses the ``"elements"`` key from the loaded info dictionary to determine
the full set of elements used by the test, and defines a function to filter the table
metrics based on a list of elements.

Partial filtering will require custom filter functions for each application, and is why
we store elemental information per data point, rather than simply the combined set.
