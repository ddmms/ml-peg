======================
Command line interface
======================

To help run calculations, analysis, and the application, we provide the ``ml_peg``
command line tool, which is installed with the package. This provides the following
commands::

    ml_peg app
    ml_peg calc
    ml_peg analyse
    ml_peg status
    ml_peg download
    ml_peg list


For example, to run the X23 test with mace-mp-0a and orb-v3-consv-inf-omat, you can run::

    ml_peg calc --test X23 --models mace-mp-0a,orb-v3-consv-inf-omat


A description of each subcommand, as well as valid options, can be listed using the
``--help`` option. For example::


    ml_peg calc --help

The ``ml_peg list`` command provides a further set of subcommands::


    ml_peg list calcs
    ml_peg list analysis
    ml_peg list app
    ml_peg list models


which list the available tests and categories that may be run for ``ml_peg calc``,
``ml_peg analyse`` and ``ml_peg app``, and the MLIPs that these can be run for.


Calculation status
==================

Each calculation records when it ran, and whether it completed, in a ``run_log.json``
written alongside its outputs. The ``ml_peg status`` command reports these logs,
totalling the state of each model per benchmark::

    ml_peg status --category supramolecular

Passing ``--detail`` reports each model separately, with the time it finished, how
long it took, how many warnings it raised, and why it is in the state it is. To look
only at calculations that need attention::

    ml_peg status --state failed,incomplete

Models that have never been run are left out of ``--detail``, as they would
otherwise be most of the table. To list them::

    ml_peg status --state not-run

A calculation is reported as:

``completed``
    Ran to the end. It may still have warned about individual structures, so check
    the warning count.
``failed``
    Raised an exception, which is reported alongside it.
``incomplete``
    Started but never finished, so was interrupted, by a scheduler timeout or an
    out-of-memory kill. A calculation currently running also appears this way.
``stale``
    Completed, but the model has since been redefined in the models file, so the
    results were produced by a different model.
``not run``
    Has no log, so was never started.
