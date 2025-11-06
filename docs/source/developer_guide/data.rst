====
Data
====

This guide will break down how to upload and download data needed for calculations,
analysis, and running the application.

Calculations and analysis
-------------------------

If there is a significant amount of data required to run a calculation or analyse the
results of a calculation, it should be made available to download, rather than being
added to the ML-PEG code repository.

There are currently two main options for storing this data: (1) within a different GitHub
repository, or (2) in an S3 bucket that we provide.

1. Data stored in GitHub

Data to be stored in a GitHub repository must be uploaded manually, but the
``download_github_data`` helper function may be used within your calculations/analysis
to access this data, by saving it to your local cache.

For example, to download a file stored at
https://github.com/joehart2001/mlipx/blob/main/benchmark_data/LNCI16_data.zip, the
following can be used:

.. code-block:: python

    from ml_peg.calcs.utils.utils import download_github_data

    data_dir = download_github_data(
        filename="LNCI16_data.zip",
        github_uri="https://raw.githubusercontent.com/joehart2001/mlipx/main/benchmark_data/",
    )

This function automatically tries to unzip zipped files, and returns the ``Path`` to
the cache directory that the file is downloaded to.

2. Data stored in S3

We provide an S3 bucket, http://s3.echo.stfc.ac.uk/ml-peg-data/, to which data can be
uploaded and downloaded.

Data can be uploaded using the ``ml_peg`` CLI, if you have the appropriate access
credentials. Input files for calculations should be stored as
``inputs/[category]/[benchmark]/[filename]`` using the ``--key`` option. For example,
to upload ``S24.zip``:

.. code-block:: bash

    ml_peg upload --key inputs/surfaces/S24/S24.zip  --filename S24.zip --credentials credentials.json

By default, this uploads to the ``https://s3.echo.stfc.ac.uk`` endpoint, into the
``ml-peg-data`` bucket

This data can be downloaded similarly:

.. code-block:: bash

    ml_peg download --key inputs/surfaces/S24/S24.zip  --filename S24.zip


Within calculations/analysis, the ``download_s3_data`` function can be used to access
this data:

.. code-block:: python

    from ml_peg.calcs.utils.utils import download_github_data

    data_dir = download_s3_data(filename="S24.zip", key="inputs/surfaces/S24/S24.zip")


Similarly to ``download_github_data``, this function automatically tries to unzip
zipped files, and returns the ``Path`` to the cache directory that the file is
downloaded to.


Application
-----------

As discussed in the :doc:`user guide </user_guide/get_started>`,
a compressed zip file containing all of the data required to run the live version of
the application is available to download from https://s3.echo.stfc.ac.uk/ml-peg-data.

When new benchmarks are added, this should be updated:

.. code-block:: bash

    ml_peg upload --key app/data/data.tar.gz  --filename data.tar.gz --credentials credentials.json
