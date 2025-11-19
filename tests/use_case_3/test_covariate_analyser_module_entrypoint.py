"""Testing the entrypoint."""

import sys
from unittest.mock import patch


def test_main(tmp_path) -> None:
    from use_case_3.covariate_analyser_module_entrypoint import app

    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_preprocessed_timeseries.zarr"
    output_path = tmp_path / "output.json"

    with patch.object(
        sys,
        "argv",
        [
            "covariate_analyser_module",
            "--input_path",
            input_zarr_path,
            "--output_path",
            str(output_path),
        ],
    ):
        assert not output_path.exists()
        app(result_action="return_value")
        assert output_path.is_file()
