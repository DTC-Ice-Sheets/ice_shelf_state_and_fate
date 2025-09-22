"""Module entrypoint for the timeseries pre-processor module."""

import argparse
from types import NoneType, UnionType
from typing import get_args, get_origin

import xarray as xr

from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.schemas import TimeseriesPreprocessorModuleEntryPointSchema
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesRegion,
    TimeseriesType,
)


def main() -> None:
    """Execute the timeseries pre-processor module from CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="use_case_3.timeseries_preprocessor_module",
        description="Pre-process a list of timeseries for covariate analysis.",
    )

    # Add arguments to the parser
    for name, field in TimeseriesPreprocessorModuleEntryPointSchema.model_fields.items():
        # The type argument needs to be a cast function
        type_ = field.annotation
        default = field.default
        extra_args = {}
        # If `None`, just default to int
        if type_ is None:
            type_ = int
        # If list, set to element type, and add `append` to parameter
        if get_origin(type_) is list:
            type_ = get_args(type_)[0]
            default = []
            extra_args["action"] = "append"
        if isinstance(type_, UnionType) and get_args(type_)[1] is NoneType:
            type_ = get_args(type_)[0]

        parser.add_argument(
            f"--{name}",
            dest=name,
            type=type_,
            default=default,
            help=field.description,
            required=name in ["input_paths", "output_path"],
            **extra_args,
        )

    # Parse the input arguments
    args = parser.parse_args()

    model = TimeseriesPreprocessorModuleEntryPointSchema(**vars(args))

    timeseries = []
    xarrays = {}
    for path in model.input_paths:
        ds = xr.open_zarr(path)
        ts = Timeseries(
            label=ds.attrs["label"],
            timeseries_type=TimeseriesType(ds.attrs["timeseries_type"]),
            timeseries_region=TimeseriesRegion(ds.attrs["timeseries_region"]),
            measurement_name=ds.attrs["measurement_name"],
            measurements=ds["measurement"].to_series(),
            measurement_std=ds["std"].to_series(),
            measurement_n=ds["n"].to_series(),
            **(
                {
                    "measurement_uncertainties": ds["uncertainty"].to_series(),
                    "measurement_uncertainties_std": ds["uncertainty_std"].to_series(),
                    "measurement_uncertainties_n": ds["uncertainty_n"].to_series(),
                }
                if "uncertainty" in ds
                else {}
            ),
        )
        timeseries.append(ts)
        xarrays[ts.label] = ds

    preprocessor = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=AggregationPeriod(model.time_period),
        aggregation=AggregationStrategy(model.aggregation),
    )

    data_vars = {}
    for ts_name, ts in preprocessor.timeseries.items():
        data_vars[f"{ts_name}_{ts.measurement_name}"] = (["time"], ts.measurements.values)
        data_vars[f"{ts_name}_{ts.measurement_name}_uncertainty"] = (["time"], ts.measurement_uncertainties.values)

    output = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": next(iter(preprocessor.timeseries.values())).measurements.index.values,
        },
    )
    output.to_zarr(model.output_path, mode="w")


if __name__ == "__main__":
    main()
