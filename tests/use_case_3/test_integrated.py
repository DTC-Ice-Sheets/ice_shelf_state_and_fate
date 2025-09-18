"""Integration test for use_case_3 module entrypoints."""

import sys
from unittest.mock import patch


def test_integrated(tmp_path) -> None:
    from use_case_3.timeseries_module_entrypoint import main as timeseries_main
    from use_case_3.timeseries_preprocessor_module_entrypoint import (
        main as preprocessor_main,
    )

    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_dataset.zarr"
    timeseries_path = tmp_path / "timeseries.zarr"
    preprocessed_timeseries_path = tmp_path / "preprocessed.zarr"

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
            str(timeseries_path),
            "--measurement_name",
            "measurement",
            "--measurement_uncertainty_name",
            "measurement_uncertainties",
            "--uncertainty",
            "std",
        ],
    ):
        assert not timeseries_path.exists()
        timeseries_main()
        assert timeseries_path.is_dir()

    with patch.object(
        sys,
        "argv",
        [
            "timeseries_preprocessor_module",
            "--input_paths",
            str(timeseries_path),
            "--output_path",
            str(preprocessed_timeseries_path),
        ],
    ):
        assert not preprocessed_timeseries_path.exists()
        preprocessor_main()
        assert preprocessed_timeseries_path.is_dir()
