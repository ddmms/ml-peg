======================
Command line interface
======================

To help run calculations, analysis, and the application, we provide the ``ml_peg``
command line tool, which is installed with the package. This provides the following
commands::

    ml_peg app
    ml_peg calc
    ml_peg analyse
    ml_peg download
    ml_peg collect
    ml_peg distribute
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


Moving outputs from an HPC
--------------------------

``ml_peg collect`` and ``ml_peg distribute`` move calculation outputs between
machines, for example to run calculations on an HPC and then analyse them or host
the app locally.

On the HPC, ``ml_peg collect`` copies every ``outputs*`` directory into one bundle
that mirrors ``ml_peg/calcs/<category>/<benchmark>/``. The ``--category``,
``--test`` and ``--models`` options select what to collect, as for ``ml_peg calc``.
Files directly inside an outputs directory, outside any model folder, are always
included::

    ml_peg collect --models mace-mp-0a --archive

This writes ``ml_peg_outputs.tar.gz``, or an ``ml_peg_outputs`` folder without
``--archive``. A single archive is quicker to copy than many small files, while a
folder suits ``rsync``. ``--output`` sets a different name.

After copying the bundle to your local machine, ``ml_peg distribute`` copies it into
your local ``ml_peg/calcs``::

    scp hpc:path/to/ml_peg_outputs.tar.gz .
    ml_peg distribute ml_peg_outputs.tar.gz

Existing local files are never overwritten. Each skipped file is counted in the
summary printed for its model, so to replace a model's results, delete its outputs
folder before distributing. Outputs for benchmarks that are not in your local
checkout are reported and skipped.
