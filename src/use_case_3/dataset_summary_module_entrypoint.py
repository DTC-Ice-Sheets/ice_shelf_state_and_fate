"""Entrypoint for the dataset summary module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cyclopts
import xarray as xr

from use_case_3.dataset_summary_module.dataset_summary_module import calculate_summary_grid
from use_case_3.helper_functions import ZarrType


def _json_encoder(obj: Any) -> Any:  # noqa: ANN401
    if isinstance(obj, datetime):
        return int(obj.timestamp())
    return obj


app = cyclopts.App()


@app.default
def main(
    input_path: ZarrType,
    output_path: Path,
    max_res: int = 20,
) -> None:
    """Process a GeoZarr dataset into a summary.

    Parameters
    ----------
    input_path : ZarrType
        Path to the input timeseries .zarr file. Can be a local path or web URL.
    output_path : Path
        Path to save the output .json file.
    max_res : int, optional
        Maximum resolution for the dataset summary.
    """
    data = xr.open_zarr(input_path)
    summary = calculate_summary_grid(data, max_res=max_res)
    with output_path.open("w") as f:
        json.dump(summary, f, default=_json_encoder)


if __name__ == "__main__":
    app()
