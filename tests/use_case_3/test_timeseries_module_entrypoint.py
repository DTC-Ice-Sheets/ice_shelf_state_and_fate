"""Testing the entrypoint."""

import json
import sys
from unittest.mock import patch


def test_main(tmp_path) -> None:
    from use_case_3.dataset_summary_module_entrypoint import app

    input_zarr_path = "tests/use_case_3/test_inputs/rasters/test_dataset_A.zarr"
    output_path = tmp_path / "output.json"

    with patch.object(
        sys,
        "argv",
        [
            "dataset_summary_module",
            "--input_path",
            input_zarr_path,
            "--output_path",
            str(output_path),
            "--max_res",
            "5",
        ],
    ):
        assert not output_path.exists()
        app(result_action="return_value")
        assert output_path.is_file()
        with output_path.open("r") as f:
            out = json.load(f)
        assert out["dims"].keys() == {"x", "y", "time"}
        assert out["vars"].keys() == {"measurement", "measurement_uncertainties"}
        assert all(len(out["dims"][dim]["idxs"]) == 5 for dim in out["dims"])
        assert all(len(out["dims"][dim]["values"]) == 5 for dim in out["dims"])
