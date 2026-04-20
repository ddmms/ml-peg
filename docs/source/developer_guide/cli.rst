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
    ml_peg info


For example, to run the X23 test with mace-mp-0a and orb-v3-consv-inf-omat, you can run::

    ml_peg calc --test X23 --models mace-mp-0a,orb-v3-consv-inf-omat


A description of each subcommand, as well as valid options, can be listed using the
``--help`` option. For example::


    ml_peg calc --help

The ``ml_peg info`` command provides a further set of subcommands::


    ml_peg info calc
    ml_peg info analysis
    ml_peg info app
    ml_peg info models


which list the available tests and categories that may be run for ``ml_peg calc``,
``ml_peg analyse`` and ``ml_peg app``, and the MLIPs that these can be run for.
