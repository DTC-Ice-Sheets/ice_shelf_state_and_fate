"""Tests for dataset summary module in use case 3."""

import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr

from use_case_3.timeseries_zarr_to_json_module.timeseries_zarr_to_json_module import timeseries_zarr_to_json


@pytest.mark.parametrize(
    "module_name",
    [
        "use_case_3.timeseries_zarr_to_json_module.timeseries_zarr_to_json_module",
        "use_case_3.timeseries_zarr_to_json_module_entrypoint",
    ],
)
def test_import_outside_repository_root(module_name, tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module_name!r})"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_summary_timeseries_zarr_to_json():
    data_xr = xr.open_dataset(Path("tests/use_case_3/test_inputs/rasters/test_timeseries_calving.zarr"))

    out_dict = timeseries_zarr_to_json(data=data_xr)

    assert out_dict == {
        "calving": {
            "measurements": [0, 1, 2],
            "uncertainties": [3, 4, 5],
            "time": [6, 7, 8],
            "timeseries_type": "derivative",
            "uncertainty_type": "std",
            "units": "Gt yr-1",
        }
    }
