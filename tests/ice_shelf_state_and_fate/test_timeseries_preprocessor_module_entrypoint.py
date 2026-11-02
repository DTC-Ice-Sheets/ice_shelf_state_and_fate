"""Testing the entrypoint."""

import re
import sys
from pathlib import Path
from unittest.mock import patch

import cyclopts
import pytest

from ice_shelf_state_and_fate.timeseries_preprocessor_module_entrypoint import app


def test_main_valid_paths(tmp_path) -> None:
    input_zarr_paths = [
        "tests/ice_shelf_state_and_fate/test_inputs/rasters/test_timeseries_A.zarr",
        "tests/ice_shelf_state_and_fate/test_inputs/rasters/test_timeseries_B.zarr",
    ]
    output_zarr_path = tmp_path / "output.zarr"

    argv = ["timeseries_preprocessor_module"]
    for input_zarr_path in input_zarr_paths:
        argv += ["--input_paths", input_zarr_path]
    argv += ["--output_path", str(output_zarr_path)]

    with patch.object(sys, "argv", argv):
        assert not output_zarr_path.exists()
        app(result_action="return_value")
        assert output_zarr_path.is_dir()


@pytest.mark.parametrize(
    ("input_path_or_urls", "patch_targets"),
    [
        # single url path
        (("https://example.com/test_timeseries_A.zarr",), ("timeseries_a_zarr",)),
        # multiple url paths
        (
            ("https://example.com/test_timeseries_A.zarr", "https://example.com/test_timeseries_B.zarr"),
            ("timeseries_a_zarr", "timeseries_b_zarr"),
        ),
        # mixture of local and url paths
        (
            (
                "tests/ice_shelf_state_and_fate/test_inputs/rasters/test_timeseries_A.zarr",
                "https://example.com/test_timeseries_B.zarr",
            ),
            ("timeseries_a_zarr", "timeseries_b_zarr"),
        ),
    ],
)
def test_main_valid_input_url(
    tmp_path: Path, input_path_or_urls: tuple[str, ...], patch_targets: tuple[str, ...], request
) -> None:
    input_zarr_paths = input_path_or_urls
    patch_target_list = [request.getfixturevalue(target) for target in patch_targets]
    output_zarr_path = tmp_path / "output.zarr"

    argv = ["timeseries_preprocessor_module"]
    for input_zarr_path in input_zarr_paths:
        argv += ["--input_paths", input_zarr_path]
    argv += ["--output_path", str(output_zarr_path)]

    with (
        patch.object(sys, "argv", argv),
        patch(
            "ice_shelf_state_and_fate.timeseries_preprocessor_module_entrypoint.xr.open_zarr",
            side_effect=patch_target_list,
        ),
    ):
        assert not output_zarr_path.exists()
        app(result_action="return_value")
        assert output_zarr_path.is_dir()


@pytest.mark.parametrize(
    "test_sources",
    [["invalid_file_path"], ["invalid_path"], ["invalid_url"]],
)
def test_main_invalid_input_raises_validation_error(
    test_sources: list[str], tmp_zarr_output_path: Path, request
) -> None:
    argv = ["timeseries_preprocessor_module"]
    for input_zarr_path in test_sources:
        argv += ["--input_paths", str(request.getfixturevalue(input_zarr_path))]
    argv += ["--output_path", str(tmp_zarr_output_path)]
    # The first input path argv will trigger the validation error.
    # We expect the first input path at index 2 of argv = [<script name>, '--input_paths', <first input path>, ...]
    expected_error_path = argv[2]

    with (
        patch.object(sys, "argv", argv),
        pytest.raises(
            cyclopts.ValidationError,
            match=re.escape(f"The provided path '{expected_error_path}' is neither a valid local path nor a web URL."),
        ),
    ):
        app(result_action="return_value", exit_on_error=False)
