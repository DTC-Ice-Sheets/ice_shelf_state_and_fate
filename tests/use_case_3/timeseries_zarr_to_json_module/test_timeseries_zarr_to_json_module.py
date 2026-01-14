"""Tests for dataset summary module in use case 3."""

from pathlib import Path

import xarray as xr

from use_case_3.timeseries_zarr_to_json_module.timeseries_zarr_to_json_module import timeseries_zarr_to_json


def test_summary_timeseries_zarr_to_json():
    data_xr = xr.open_dataset(Path("tests/use_case_3/test_inputs/rasters/test_timeseries_calving.zarr"))

    out_dict = timeseries_zarr_to_json(data=data_xr)

    assert out_dict["calving"]["measurements"] == [0, 1, 2]
    assert out_dict["calving"]["uncertainties"] == [3, 4, 5]
    assert out_dict["calving"]["time"] == [6, 7, 8]
    assert out_dict["calving"]["timeseries_type"] == "derivative"
    assert out_dict["calving"]["uncertainty_type"] == "std"
    assert out_dict["calving"]["units"] == "Gt yr-1"

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
