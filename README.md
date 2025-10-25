# ML-PEG

[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![License][license-badge]][license-link]
[![DOI][doi-badge]][doi-link]

ML potential usability and performance guide

> [!NOTE]
> Migration in progress! The live benchmarks are currently run and analysed using
> [mlipx](https://github.com/basf/mlipx) nodes defined in this repository:
> https://github.com/joehart2001/mlipx.
>
> New benchmarks are expected to be added following the format defined in this
> repository, and work is ongoing to migrate all existing benchmarks to this format.

Our original interactive analysis suite is currently hosted at: http://mlip-testing.stfc.ac.uk:8050

## Contents
- [Getting started](#getting-started)
- [Features](#features)
- [Docker/Podman images](#dockerpodman-images)
- [Development](#development)
- [License](#license)

## Getting started

### Dependencies

All required and optional dependencies can be found in [pyproject.toml](pyproject.toml).


### Installation

The latest stable release of ML-PEG, including its dependencies, will be installable from PyPI by running:

```
python3 -m pip install ml-peg
```

To get all the latest changes, ML-PEG can be installed from GitHub:

```
python3 -m pip install git+https://github.com/ddmms/ml-peg.git
```


## Features

Coming soon!


## Docker/Podman images

You can use [Docker](https://www.docker.com) or [Podman](https://podman.io/) to build
and/or run the ML-PEG app yourself.

> [!TIP]
> The commands below will assume you are using Docker. To use Podman, replace `docker`
> with `podman`, e.g. `podman pull`, `podman build`, and `podman run`.

A Docker image with the latest changes can be pulled from the
GitHub container registry, following the command that can be found under this
repository's [packages](https://github.com/ddmms/ml-peg/pkgs/container/ml-peg-app).


> [!NOTE]
> Currently, this repository only contains images for the linux/amd64 platform.
> On MacOS with ARM silicon, this can often still be run by setting
> `--platform linux/amd64` when using `docker run`.


Alternatively, to build the container yourself, you can use the
[Dockerfile](containers/Dockerfile) provided. From the `ml-peg` directory, run:

```
docker build -t ml-peg-app -f containers/Dockerfile .
```

Once built, you can mount your current application data and start the app by running:

```
docker run --volume ./ml_peg/app/data:/app/ml_peg/app/data  --publish 8050:8050 ml-peg-app
```

> [!TIP]
> Ensure `ml_peg/app/data` is populated with results before running the container.


Alternatively, you can use the [compose.yml](containers/compose.yml) file provided, via
Docker Compose:

```
docker compose -f containers/compose.yml up -d
```

The app should now be accessible at http://localhost:8050.

## Development

Please ensure you have consulted our
[contribution guidelines](contributing.md)
and
[coding style](coding_style.md)
before proceeding.

We recommend installing `uv` for dependency management when developing for ML-PEG:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation)
2. Install ML-PEG with dependencies in a virtual environment:

```shell
git clone https://github.com/ddmms/ml-peg
cd ml-peg
uv sync # Create a virtual environment and install dependencies
source .venv/bin/activate
pre-commit install  # Install pre-commit hooks
pytest -v  # Discover and run all tests
```

Please refer to the [online documentation](https://ddmms.github.io/ml-peg/developer_guide/index.html)
for information about contributing new benchmarks and models.

## License

[GNU General Public License version 3](LICENSE)

[ci-badge]: https://github.com/ddmms/ml-peg/actions/workflows/ci.yml/badge.svg?branch=main
[ci-link]: https://github.com/ddmms/ml-peg/actions
[cov-badge]: https://coveralls.io/repos/github/ddmms/ml-peg/badge.svg?branch=main
[cov-link]: https://coveralls.io/github/ddmms/ml-peg?branch=main
[docs-badge]: https://github.com/ddmms/ml-peg/actions/workflows/docs.yml/badge.svg
[docs-link]: https://ddmms.github.io/ml-peg/
[license-badge]: https://img.shields.io/badge/License-GPLv3-blue.svg
[license-link]: https://opensource.org/license/gpl-3-0
[doi-link]: https://doi.org/10.5281/zenodo.16904445
[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.16904445.svg
