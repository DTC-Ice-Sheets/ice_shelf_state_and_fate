"""Tests for dataset summary module in use case 3."""


from pathlib import Path

from use_case_3.summary_data_module.summary_data import summary_timeseries_zarr_to_json


def test_summary_timeseries_zarr_to_json():
    input_zarr_path = Path("tests/use_case_3/test_inputs/rasters/test_timeseries_calving.zarr")

    out_dict = summary_timeseries_zarr_to_json(input_zarr_path)

    assert out_dict["calving"]["measurements"] == [0, 1, 2]
    assert out_dict["calving"]["uncertainties"] == [3, 4, 5]
    assert out_dict["calving"]["time"] == [6, 7, 8]
    assert out_dict["calving"]["timeseries_type"] == "derivative"
    assert out_dict["calving"]["uncertainty_type"] == "std"
    assert out_dict["calving"]["units"] == "Gt yr-1"
