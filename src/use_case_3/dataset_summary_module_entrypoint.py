"""Entrypoint for the dataset summary module."""

import argparse
import json
from datetime import datetime
from types import UnionType
from typing import Any, get_args

import xarray as xr

from use_case_3.dataset_summary_module.dataset_summary_module import calculate_summary_grid
from use_case_3.schemas import DatasetSummaryEntrypointSchema


def _json_encoder(obj: Any) -> Any:  # noqa: ANN401
    if isinstance(obj, datetime):
        return int(obj.timestamp())
    return obj


def main() -> None:
    """Execute the dataset summary module from CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="use_case_3.dataset_summary_module",
        description="Process a GeoZarr dataset into a summary.",
    )

    # Add arguments to the parser
    for name, field in DatasetSummaryEntrypointSchema.model_fields.items():
        # The type argument needs to be a function, so remove ` | None` if present
        type_ = field.annotation
        if type_ is None:
            type_ = int
        elif isinstance(type_, UnionType):
            type_ = get_args(type_)[0]

        parser.add_argument(
            f"--{name}",
            dest=name,
            type=type_,
            default=field.default,
            help=field.description,
            required=name in ["label", "dataset_path", "output_path", "measurement_name"],
        )

    # Parse the input arguments
    args = parser.parse_args()

    # Create the model from parsed arguments
    model = DatasetSummaryEntrypointSchema(**vars(args))

    data = xr.open_zarr(model.input_path)
    summary = calculate_summary_grid(data, max_res=model.max_res)
    with model.output_path.open("w") as f:
        json.dump(summary, f, default=_json_encoder)


if __name__ == "__main__":
    main()
