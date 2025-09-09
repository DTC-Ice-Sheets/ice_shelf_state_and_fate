"""Entrypoint for timeseries module."""

import argparse
import json
import logging
import xarray as xr
from types import UnionType
from typing import get_args

from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.schemas import TimeseriesModuleEntryPointSchema

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the timeseries module from CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="use_case_3.timeseries_module",
        description="Process a GeoZarr dataset into a timeseries.",
    )

    # Add arguments to the parser
    for name, field in TimeseriesModuleEntryPointSchema.model_fields.items():
        # The type argument needs to be a function, so remove ` | None` if present
        _type = field.annotation
        if _type is None:
            _type = int
        elif isinstance(_type, UnionType):
            _type = get_args(_type)[0]

        parser.add_argument(
            f"--{name}",
            dest=name,
            type=_type,
            default=field.default,
            help=field.description,
            required=name in ["label", "dataset_path", "output_path", "measurement_name"],
        )

    # Parse the input arguments
    args = parser.parse_args()

    # Create the model from parsed arguments
    model = TimeseriesModuleEntryPointSchema(**vars(args))

    timeseries = Timeseries(
        label=model.label,
        timeseries_type=model.timeseries_type,
        timeseries_region=model.timeseries_region,
    )

    data = xr.open_zarr(model.dataset_path)
    if model.masks_json_path is not None:
        with model.masks_json_path.open("r") as f:
            mask_geometries = json.load(f)
    else:
        mask_geometries = {"ice_shelf": None, "ocean": None, "climate": None}

    timeseries.process_timeseries(
        data=data,
        measurement_name=model.measurement_name,
        measurement_uncertainty_name=model.measurement_uncertainty_name,
        mask_geometries=mask_geometries,
        apply_mask=model.apply_mask,
        aggregate=model.aggregate,
        period=model.period,
        uncertainty=model.uncertainty,
        independent=model.independent,
        t0_normalisation=model.t0_normalisation,
    )

    output = xr.Dataset(
        data_vars={
            "measurement": (["time"], timeseries.measurements.values),
            "uncertainty": (
                (["time"], timeseries.measurement_uncertainties.values)
                if timeseries.measurement_uncertainties is not None
                else None
            ),
        },
        coords={
            "time": timeseries.measurements.index.values,
        },
    )

    output.to_zarr(model.output_path, mode="w")


if __name__ == "__main__":
    main()
