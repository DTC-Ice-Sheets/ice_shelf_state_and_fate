"""Tests for timeseries zarr to json module in the 'Ice Shelf State and Fate' package."""

import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr

from ice_shelf_state_and_fate.timeseries_zarr_to_json_module.timeseries_zarr_to_json_module import (
    timeseries_zarr_to_json,
)


@pytest.mark.parametrize(
    "module_name",
    [
        "ice_shelf_state_and_fate.timeseries_zarr_to_json_module.timeseries_zarr_to_json_module",
        "ice_shelf_state_and_fate.timeseries_zarr_to_json_module_entrypoint",
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
    data_xr = xr.open_dataset(Path("tests/ice_shelf_state_and_fate/test_inputs/rasters/test_timeseries_calving.zarr"))

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
