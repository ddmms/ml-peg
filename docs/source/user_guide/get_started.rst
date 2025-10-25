===============
Getting started
===============

Dependencies
------------

All required and optional dependencies can be found in `pyproject.toml <https://github.com/ddmms/ml-peg/blob/main/pyproject.toml>`_.


Installation
------------

The latest stable release of ML-PEG, including its dependencies, will soon be installable from PyPI by running:

.. code-block:: bash

    python3 -m pip install ml-peg


To get all the latest changes, ML-PEG can also be installed from GitHub:

.. code-block:: bash

    python3 -m pip install git+https://github.com/ddmms/ml-peg.git


Running the application
-----------------------

You can use `Docker <https://www.docker.com>`_ or `Podman <https://podman.io/>`_ to
build and/or run the ML-PEG app yourself.

.. tip::

    The commands below will assume you are using Docker. To use Podman, replace
    ``docker`` with ``podman``, e.g. ``podman pull``, ``podman build``, and
    ``podman run``.


A Docker image with the latest changes can be pulled from the GitHub container
registry, following the command that can be found under this repository's
`packages <https://github.com/ddmms/ml-peg/pkgs/container/ml-peg-app>`_.

.. note::

    Currently, this repository only contains images for the linux/amd64 platform.
    On MacOS with ARM silicon, this can often still be run by setting
    ``--platform linux/amd64`` when using ``docker run``.


Alternatively, to build the container yourself, you can use the
`Dockerfile <https://github.com/ddmms/ml-peg/blob/main/containers/Dockerfile>`_
provided. From the ``ml-peg`` directory, run:

.. code-block:: bash

    docker build -t ml-peg-app -f containers/Dockerfile .


Once built, you can mount your current application data and start the app by running:

.. code-block:: bash

    docker run --volume ./ml_peg/app/data:/app/ml_peg/app/data  --publish 8050:8050 ml-peg-app

.. tip::

    Ensure ``ml_peg/app/data`` is populated with results before running the container.


Alternatively, you can use the
`compose.yml <https://github.com/ddmms/ml-peg/blob/main/containers/compose.yml>`_
file provided, via Docker Compose:

.. code-block:: bash

    docker compose -f containers/compose.yml up -d


The app should now be accessible at http://localhost:8050.
