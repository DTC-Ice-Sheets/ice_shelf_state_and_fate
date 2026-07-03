"""Testing the entrypoint."""

import re
import sys
from pathlib import Path
from unittest.mock import patch

import cyclopts
import pytest
from xarray import Dataset

from use_case_3.timeseries_zarr_to_json_module_entrypoint import app


def test_main_valid_path(tmp_json_output_path) -> None:
    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_preprocessed_timeseries.zarr"

    with patch.object(
        sys,
        "argv",
        [
            "timeseries_zarr_to_json_module",
            "--input_path",
            input_zarr_path,
            "--output_path",
            str(tmp_json_output_path),
        ],
    ):
        assert not tmp_json_output_path.exists()
        app(result_action="return_value")
        assert tmp_json_output_path.is_file()


def test_main_valid_input_url(preprocessed_timeseries_zarr: Dataset, tmp_json_output_path: Path) -> None:
    input_zarr_path = "https://example.com/test_preprocessed_timeseries.zarr"

    with (
        # to avoid making an actual HTTP request, we patch the open_zarr function to return our fixture dataset
        patch(
            "use_case_3.timeseries_zarr_to_json_module_entrypoint.xr.open_dataset",
            return_value=preprocessed_timeseries_zarr,
        ),
        patch.object(
            sys,
            "argv",
            [
                "timeseries_zarr_to_json_module",
                "--input_path",
                input_zarr_path,
                "--output_path",
                str(tmp_json_output_path),
            ],
        ),
    ):
        assert not tmp_json_output_path.exists()
        app(result_action="return_value")
        assert tmp_json_output_path.is_file()


@pytest.mark.parametrize(
    "test_source",
    ["invalid_file_path", "invalid_path", "invalid_url"],
)
def test_main_invalid_input_raises_validation_error(test_source: str, tmp_json_output_path: Path, request) -> None:
    error_string = test_data = request.getfixturevalue(test_source)
    with (
        patch.object(
            sys,
            "argv",
            [
                "timeseries_zarr_to_json_module",
                "--input_path",
                str(test_data),
                "--output_path",
                str(tmp_json_output_path),
            ],
        ),
        pytest.raises(
            cyclopts.ValidationError,
            match=re.escape(f"The provided path '{error_string}' is neither a valid local path nor a web URL."),
        ),
    ):
        app(result_action="return_value", exit_on_error=False)
