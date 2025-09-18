"""Testing the entrypoint."""

import sys
from unittest.mock import patch


def test_main(tmp_path) -> None:
    from use_case_3.timeseries_preprocessor_module_entrypoint import main

    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_timeseries.zarr"
    output_zarr_path = tmp_path / "output.zarr"

    with patch.object(
        sys,
        "argv",
        [
            "timeseries_preprocessor_module",
            "--input_paths",
            input_zarr_path,
            "--output_path",
            str(output_zarr_path),
        ],
    ):
        assert not output_zarr_path.exists()
        main()
        assert output_zarr_path.is_dir()
