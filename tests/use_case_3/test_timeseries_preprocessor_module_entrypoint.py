"""Testing the entrypoint."""

import sys
from unittest.mock import patch

from use_case_3.timeseries_preprocessor_module_entrypoint import app


def test_main(tmp_path) -> None:

    input_zarr_paths = [
        "tests/use_case_3/test_inputs/rasters/test_timeseries_A.zarr",
        "tests/use_case_3/test_inputs/rasters/test_timeseries_B.zarr",
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
