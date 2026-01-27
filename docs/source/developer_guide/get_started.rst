===============
Getting started
===============

Please ensure you have consulted our
`contribution guidelines <https://github.com/ddmms/ml-peg/blob/main/contributing.md>`_
and
`coding style <https://github.com/ddmms/ml-peg/blob/main/coding_style.md>`_
before proceeding.


Dependency management
---------------------

We recommend `installing uv <https://docs.astral.sh/uv/getting-started/installation/>`_
for dependency management when developing for ML-PEG.

This provides a number of useful features, including:

- `Dependency management <https://docs.astral.sh/uv/concepts/projects/dependencies/>`_ (``uv [add,remove]`` etc.) and organization (`groups <https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups>`_)

- Storing the versions of all installations in a `uv.lock <https://docs.astral.sh/uv/concepts/projects/sync/>`_ file, for reproducible builds

- Improved `dependency resolution <https://docs.astral.sh/uv/concepts/resolution/>`_

- Virtual environment management

- `Building and publishing <https://docs.astral.sh/uv/guides/publish/>`_ tools


Dependencies useful for development can then be installed by running::

    uv sync -p 3.12
    source .venv/bin/activate


`Extras <https://peps.python.org/pep-0508/#extras>`_ are used to install MLIP
packages, as well dispersion. These are defined in the
``[project.optional-dependencies]`` section of ``pyproject.toml``, and can be installed
by adding ``--extra``. For example, to install MACE, Orb, and torch-dftd, you can run::

    uv sync --extra mace --extra orb --extra d3


Using uv
--------

``uv`` manages a `persistent environment <https://docs.astral.sh/uv/concepts/projects/layout/#the-project-environment>`_
with the project and its dependencies in a ``.venv`` directory, adjacent to ``pyproject.toml``. This will be created automatically as needed.

``uv`` provides two separate APIs for managing your Python project and environment.

``uv pip`` is designed to resemble the ``pip`` CLI, with similar commands (``uv pip install``,  ``uv pip list``, ``uv pip tree``, etc.),
and is slightly lower level. `Compared with pip <https://docs.astral.sh/uv/pip/compatibility/>`_,
``uv`` tends to be stricter, but in most cases ``uv pip`` could be used in place of ``pip``.

``uv add``, ``uv run``, ``uv sync``, and ``uv lock`` are known as "project APIs", and are slightly higher level.
These commands interact with (and require) ``pyproject.toml``, and ``uv`` will ensure your environment is in-sync when they are called,
including creating or updating a `lockfile <https://docs.astral.sh/uv/concepts/projects/sync/>`_,
a universal resolution that is `portable across platforms <https://docs.astral.sh/uv/concepts/resolution/#universal-resolution>`_.

When developing for ML-PEG, it is usually recommended to use project commands, as described in `Getting started`_
rather than using ``uv pip install`` to modify the project environment manually.

.. tip::

    ``uv`` will detect and use Python versions available on your system,
    but can also be used to `install Python automtically <https://docs.astral.sh/uv/guides/install-python/>`_.
    The desired Python version can be specified when running project commands with the ``--python``/``-p`` option.


For further information, please refer to the `uv documentation <https://docs.astral.sh/uv/>`_.


Running unit tests
------------------

Packages in the ``dev`` dependency group allow tests to be run locally using ``pytest``, by running::

    pytest -v


Automatic coding style check
----------------------------

Packages in the ``pre-commit`` dependency group allow automatic code formatting and linting on every commit.

To set this up, run::

    pre-commit install

After this, the `ruff linter <https://docs.astral.sh/ruff/linter/>`_, `ruff formatter <https://docs.astral.sh/ruff/formatter/>`_, and `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ (docstring style validator), will run before every commit.

Rules enforced by ruff are currently set up to be comparable to:

- `black <https://black.readthedocs.io>`_ (code formatter)
- `pylint <https://www.pylint.org/>`_ (linter)
- `pyupgrade <https://github.com/asottile/pyupgrade>`_ (syntax upgrader)
- `isort <https://pycqa.github.io/isort/>`_ (import sorter)
- `flake8-bugbear <https://pypi.org/project/flake8-bugbear/>`_ (bug finder)

The full set of `ruff rules <https://docs.astral.sh/ruff/rules/>`_ are specified by the ``[tool.ruff]`` sections of `pyproject.toml <https://github.com/ddmms/ml-peg/blob/main/pyproject.toml>`_.


Building the documentation
--------------------------

Packages in the ``docs`` dependency group install `Sphinx <https://www.sphinx-doc.org>`_
and other Python packages required to build ML-PEG's documentation.

It is also necessary to `install pandoc <https://pandoc.org/installing.html>`_ on your system.

Individual individual documentation pages can be edited directly::

        docs/source/index.rst
        docs/source/user_guide/index.rst
        docs/source/user_guide/get_started.rst
        docs/source/developer_guide/index.rst
        docs/source/developer_guide/get_started.rst
        docs/source/developer_guide/add_benchmarks.rst
        docs/source/apidoc/ml_peg.rst

API documentation is automatically generated from ``docs/source/apidoc/ml_peg.rst``.

``Sphinx`` can then be used to generate the html documentation::

        cd docs
        make clean; make html


Continuous integration
----------------------

ML-PEG comes with a ``.github`` folder that contains continuous integration workflows that run on every push and pull request using `GitHub Actions <https://github.com/features/actions>`_. These will:

#. Run all non-optional unit tests
#. Build the documentation
#. Check the coding style conforms by running the pre-commit described above
#. Build and publish tagged commits to PyPI
