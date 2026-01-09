"""Entrypoint for the dataset summary module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cyclopts

from use_case_3.summary_data_module.summary_data import timeseries_zarr_to_json


def _json_encoder(obj: Any) -> Any:  # noqa: ANN401
    if isinstance(obj, datetime):
        return int(obj.timestamp())
    return obj


app = cyclopts.App()


@app.default
async def main(
    ice_shelf: str,
    output_path: Path,
) -> None:
    """Collect the summary data for an ice shelf.

    Parameters
    ----------
    ice_shelf : str
        Identifier for the ice shelf to collect summary data for.
    output_path : Path
        Path to save the output .json file.
    """
    data = await timeseries_zarr_to_json(ice_shelf=ice_shelf)
    with output_path.open("w") as f:
        json.dump(data, f, default=_json_encoder)


if __name__ == "__main__":
    app()
