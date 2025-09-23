"""Helper script to setup an S3 bucket with test data.

Useful for testing the Argo workflow.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from dtc_wrapper.helpers import configure_aws_s3_client, delete_directory_from_s3, upload_directory_to_s3

load_dotenv()

client = configure_aws_s3_client()

bucket = "wrapper-testing"


def create_dataset():
    nx, ny, nt = 5, 5, 20
    return xr.Dataset(
        data_vars={
            "measurement": (("time", "y", "x"), np.random.rand(nt, ny, nx)),
            "measurement_uncertainties": (("time", "y", "x"), np.random.rand(nt, ny, nx) * 0.1),
        },
        coords={
            "time": xr.date_range("2000-01-01", periods=nt),
            "y": np.linspace(75, 77, ny),
            "x": np.linspace(-30, -20, nx),
        },
    )


delete_directory_from_s3(bucket, "use_case_3/", client)
with TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    for dataset in ["A", "B", "C"]:
        ds = create_dataset()
        ds.to_zarr(tmpdir / "test_dataset.zarr", mode="w")
        upload_directory_to_s3(
            tmpdir / "test_dataset.zarr",
            bucket,
            f"use_case_3/input_dataset_{dataset}.zarr",
            client,
        )
