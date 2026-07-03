"""Entrypoint for the dataset summary module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cyclopts
import xarray as xr

from use_case_3.helper_functions import ZarrType
from use_case_3.timeseries_zarr_to_json_module.timeseries_zarr_to_json_module import timeseries_zarr_to_json


def _json_encoder(obj: Any) -> Any:  # noqa: ANN401
    if isinstance(obj, datetime):
        return int(obj.timestamp())
    return obj


app = cyclopts.App()


@app.default
def main(
    input_path: ZarrType,
    output_path: Path,
) -> None:
    """Collect the summary data for an ice shelf.

    Parameters
    ----------
    input_path : ZarrType
        Path to the input timeseries .zarr file. Can be a local path or a web URL.
    output_path : Path
        Path to save the output .json file.
    """
    data_xr = xr.open_dataset(input_path)
    data_dict = timeseries_zarr_to_json(data=data_xr)
    with output_path.open("w") as f:
        json.dump(data_dict, f, default=_json_encoder)


if __name__ == "__main__":
    app()
