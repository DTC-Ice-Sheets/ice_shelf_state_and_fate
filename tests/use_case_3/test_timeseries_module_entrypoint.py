"""Testing the entrypoint."""

import sys
from unittest.mock import patch


def test_main(tmp_path) -> None:
    from use_case_3.timeseries_module_entrypoint import main

    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_dataset.zarr"
    output_path = tmp_path / "output.zarr"

    with patch.object(
        sys,
        "argv",
        [
            "timeseries_module",
            "--label",
            "test_label",
            "--dataset_path",
            input_zarr_path,
            "--output_path",
            str(output_path),
            "--measurement_name",
            "measurement",
            "--measurement_uncertainty_name",
            "measurement_uncertainties",
            "--uncertainty",
            "std",
        ],
    ):
        assert not output_path.exists()
        main()
        assert output_path.is_dir()
