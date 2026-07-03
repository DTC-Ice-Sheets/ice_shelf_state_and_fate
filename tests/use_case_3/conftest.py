"""Pytest fixtures that are automatically used in use_case_3 tests."""

from pathlib import Path

import pytest
import xarray as xr


@pytest.fixture
def invalid_file_path(tmp_path: Path) -> Path:
    """Provide a path to a non-existent GeoZarr store directory for testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary path provided by pytest.
    """
    return tmp_path / "non_existent_file.zarr"


@pytest.fixture
def invalid_url() -> str:
    """Provide an invalid URL for testing."""
    return "invalid_url/doesnt/start/with/http/or/https"


@pytest.fixture
def invalid_path() -> str:
    """Provide an invalid path for testing."""
    return "invalid_path/doesnt/exist"


@pytest.fixture
def preprocessed_timeseries_zarr() -> xr.Dataset:
    """Fixture to create a temporary Zarr store for testing."""
    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_preprocessed_timeseries.zarr"
    return xr.open_zarr(input_zarr_path)


@pytest.fixture
def tmp_json_output_path(tmp_path: Path) -> Path:
    """Fixture to create a temporary JSON output path for testing."""
    return tmp_path / "output.json"
