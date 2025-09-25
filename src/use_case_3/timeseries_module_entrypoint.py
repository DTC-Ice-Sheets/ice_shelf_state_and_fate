"""Entrypoint for timeseries module."""

import argparse
import json
from types import UnionType
from typing import get_args

import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from use_case_3.schemas import TimeseriesModuleEntryPointSchema
from use_case_3.timeseries_module.timeseries import Timeseries


def main() -> None:
    """Execute the timeseries module from CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="use_case_3.timeseries_module",
        description="Process a GeoZarr dataset into a timeseries.",
    )

    # Add arguments to the parser
    for name, field in TimeseriesModuleEntryPointSchema.model_fields.items():
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
    model = TimeseriesModuleEntryPointSchema(**vars(args))

    timeseries = Timeseries(
        label=model.label,
        timeseries_type=model.timeseries_type,
        timeseries_region=model.timeseries_region,
    )

    data = xr.open_zarr(model.dataset_path)
    mask_geometries: dict[str, Polygon | MultiPolygon | None]
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
        t0_normalisation=str(model.t0_normalisation) if model.t0_normalisation is not None else None,
    )

    output = xr.Dataset(
        data_vars={
            "measurement": (["time"], timeseries.measurements.values),
            "std": (["time"], timeseries.measurement_std.values),
            "n": (["time"], timeseries.measurement_n.values),
            **(
                {
                    "uncertainty": (["time"], timeseries.measurement_uncertainties.values),
                    "uncertainty_std": (["time"], timeseries.measurement_uncertainties_std.values),
                    "uncertainty_n": (["time"], timeseries.measurement_uncertainties_n.values),
                }
                if model.measurement_uncertainty_name is not None
                else {}
            ),
        },
        coords={
            "time": timeseries.measurements.index.values,
        },
        attrs={
            "label": timeseries.label,
            "timeseries_type": timeseries.timeseries_type.value,
            "timeseries_region": timeseries.timeseries_region.value,
            "measurement_name": model.measurement_name,
            "measurement_uncertainty_name": model.measurement_uncertainty_name,
        },
    )

    output.to_zarr(model.output_path, mode="w")


if __name__ == "__main__":
    main()
