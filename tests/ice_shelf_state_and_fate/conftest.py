"""Pytest fixtures that are automatically used in use_case_3 tests."""

from pathlib import Path

import pytest
import xarray as xr


@pytest.fixture
def invalid_file_path(tmp_path: Path) -> Path:
    return tmp_path / "non_existent_file.zarr"


@pytest.fixture
def invalid_url() -> str:
    return "invalid_url/doesnt/start/with/http/or/https"


@pytest.fixture
def invalid_path() -> str:
    """Provide an invalid path for testing."""
    return "invalid_path/doesnt/exist"


@pytest.fixture
def preprocessed_timeseries_zarr() -> xr.Dataset:
    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_preprocessed_timeseries.zarr"
    return xr.open_zarr(input_zarr_path)


@pytest.fixture
def timeseries_a_zarr() -> xr.Dataset:
    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_timeseries_A.zarr"
    return xr.open_zarr(input_zarr_path)


@pytest.fixture
def timeseries_b_zarr() -> xr.Dataset:
    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_timeseries_B.zarr"
    return xr.open_zarr(input_zarr_path)


@pytest.fixture
def preprocessed_test_dataset_a_zarr() -> xr.Dataset:
    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_dataset_A.zarr"
    return xr.open_zarr(input_zarr_path)


@pytest.fixture
def tmp_json_output_path(tmp_path: Path) -> Path:
    """Fixture to create a temporary JSON output path for testing."""
    return tmp_path / "output.json"


@pytest.fixture
def tmp_zarr_output_path(tmp_path: Path) -> Path:
    """Fixture to create a temporary Zarr output path for testing."""
    return tmp_path / "output.zarr"
