"""Tests for dataset summary module in the 'Ice Shelf State and Fate' package."""

import xarray as xr

from ice_shelf_state_and_fate.dataset_summary_module.dataset_summary_module import calculate_summary_grid


def test_dataset_summary_module():
    input_zarr_path = "tests/ice_shelf_state_and_fate/test_inputs/rasters/test_dataset_A.zarr"
    ds = xr.open_zarr(input_zarr_path)

    out = calculate_summary_grid(ds, max_res=4)
    assert out["dims"].keys() == {"x", "y", "time"}
    assert out["vars"].keys() == {"measurement", "measurement_uncertainties"}
    assert all(len(out["dims"][dim]["idxs"]) == 4 for dim in out["dims"])
    assert all(len(out["dims"][dim]["values"]) == 4 for dim in out["dims"])
    assert all(out["dims"][dim]["real_size"] == ds[dim].shape[0] for dim in out["dims"])
