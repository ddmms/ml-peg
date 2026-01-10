"""Upload and download data."""

from __future__ import annotations

import json
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import tqdm


def download(
    key: str,
    filename: str,
    bucket: str = "ml-peg-data",
    endpoint: str = "https://s3.echo.stfc.ac.uk",
    credentials: str | None = None,
):
    """
    Download data from S3 bucket.

    Parameters
    ----------
    key
        Name of file in S3 bucket to download.
    filename
        Name of file to save download as locally.
    bucket
        Name of S3 bucket. Default is "ml-peg-data".
    endpoint
        Endpoint URL. Default is "https://s3.echo.stfc.ac.uk".
    credentials
        S3 credentials. Default is `None`, which will only allow downloading public
        data.
    """
    if credentials is None:
        s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED, s3={"addressing_style": "path"}),
            endpoint_url=endpoint,
        )
    else:
        with open(credentials) as credentials_file:
            user_credentials = json.load(credentials_file)

        s3 = boto3.client(
            "s3",
            config=Config(s3={"addressing_style": "path"}),
            endpoint_url=endpoint,
            aws_access_key_id=user_credentials["access_key"],
            aws_secret_access_key=user_credentials["secret_key"],
        )

    # Ensure directory exists before download
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    object_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    with tqdm.tqdm(total=object_size, unit="B", unit_scale=True, desc=filename) as pbar:
        s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=filename,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )

    # s3.download_file(Bucket=bucket, Filename=filename, Key=key)


def upload(
    key: str,
    filename: str | Path,
    credentials: str,
    bucket: str = "ml-peg-data",
    endpoint: str = "https://s3.echo.stfc.ac.uk",
    acl: str = "public-read",
):
    """
    Upload data from an S3 bucket.

    Parameters
    ----------
    key
        Name of file to create in S3 bucket.
    filename
        File to upload.
    credentials
        S3 credentials.
    bucket
        Name of S3 bucket to upload to. Default is "ml-peg-data".
    endpoint
        Endpoint URL. Default is "https://s3.echo.stfc.ac.uk".
    acl
        Access control list. Default is "public-read".
    """
    with open(credentials) as credentials_file:
        user_credentials = json.load(credentials_file)

    s3 = boto3.client(
        "s3",
        config=Config(signature_version="s3", s3={"addressing_style": "path"}),
        endpoint_url=endpoint,
        aws_access_key_id=user_credentials["access_key"],
        aws_secret_access_key=user_credentials["secret_key"],
    )

    # Ensure path exists before download
    filename = Path(filename)
    if not filename.exists():
        raise ValueError(f"{filename} does not exist")

    file_size = filename.stat().st_sizest_size
    with tqdm.tqdm(total=file_size, unit="B", unit_scale=True, desc=filename) as pbar:
        s3.upload_file(
            Filename=filename,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ACL": acl},
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )
