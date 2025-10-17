"""Integration test for use_case_3 module entrypoints."""

import sys
from unittest.mock import patch


def test_integrated(tmp_path) -> None:
    from use_case_3.covariate_analyser_module_entrypoint import (
        main as covariate_analyser_main,
    )
    from use_case_3.timeseries_module_entrypoint import main as timeseries_main
    from use_case_3.timeseries_preprocessor_module_entrypoint import (
        main as preprocessor_main,
    )

    input_zarr_path_1 = "tests/use_case_3/test_inputs/rasters/test_dataset_A.zarr"
    input_zarr_path_2 = "tests/use_case_3/test_inputs/rasters/test_dataset_B.zarr"
    timeseries_path_1 = tmp_path / "timeseries_1.zarr"
    timeseries_path_2 = tmp_path / "timeseries_2.zarr"
    preprocessed_timeseries_path = tmp_path / "preprocessed.zarr"
    analysis_output_path = tmp_path / "analysis_output.json"

    with patch.object(
        sys,
        "argv",
        [
            "timeseries_module",
            "--label",
            "test_label",
            "--dataset_path",
            input_zarr_path_1,
            "--output_path",
            str(timeseries_path_1),
            "--measurement_name",
            "measurement",
            "--measurement_uncertainty_name",
            "measurement_uncertainties",
            "--uncertainty",
            "std",
        ],
    ):
        assert not timeseries_path_1.exists()
        timeseries_main()
        assert timeseries_path_1.is_dir()

    with patch.object(
        sys,
        "argv",
        [
            "timeseries_module",
            "--label",
            "test_label",
            "--dataset_path",
            input_zarr_path_2,
            "--output_path",
            str(timeseries_path_2),
            "--measurement_name",
            "measurement",
            "--measurement_uncertainty_name",
            "measurement_uncertainties",
            "--uncertainty",
            "std",
        ],
    ):
        assert not timeseries_path_2.exists()
        timeseries_main()
        assert timeseries_path_2.is_dir()

    with patch.object(
        sys,
        "argv",
        [
            "timeseries_preprocessor_module",
            "--input_paths",
            str(timeseries_path_1),
            "--input_paths",
            str(timeseries_path_2),
            "--output_path",
            str(preprocessed_timeseries_path),
        ],
    ):
        assert not preprocessed_timeseries_path.exists()
        preprocessor_main()
        assert preprocessed_timeseries_path.is_dir()

    with patch.object(
        sys,
        "argv",
        [
            "covariate_analyser_module",
            "--input_path",
            str(preprocessed_timeseries_path),
            "--output_path",
            str(analysis_output_path),
        ],
    ):
        assert not analysis_output_path.exists()
        covariate_analyser_main()
        assert analysis_output_path.is_file()
