import argparse
from types import NoneType, UnionType
from typing import get_origin, get_args
import xarray as xr

from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.schemas import TimeseriesPreprocessorModuleEntryPointSchema
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesType,
    TimeseriesRegion,
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
        _type = field.annotation
        required = True
        extra_args = {}
        # If `None`, just default to int
        if _type is None:
            _type = int
        # If list, set to element type, and add `append` to parameter
        if get_origin(_type) is list:
            _type = get_args(_type)[0]
            extra_args["append"] = True
        if isinstance(_type, UnionType) and get_args(_type)[1] is NoneType:
            _type = get_args(_type)[0]
            required = False

        parser.add_argument(
            f"--{name}",
            dest=name,
            type=_type,
            default=field.default,
            help=field.description,
            required=required,
            append=True,
            **extra_args,
        )

    # Parse the input arguments
    args = parser.parse_args()

    model = TimeseriesPreprocessorModuleEntryPointSchema(**vars(args))

    input_paths, output_paths = zip(*(p.split(":") for p in model.timeseries))

    timeseries = []
    xarrays = []
    for path in input_paths:
        ds = xr.open_zarr(path)
        xarrays.append(ds)
        timeseries.append(
            Timeseries(
                label=ds.attrs["label"],
                timeseries_type=TimeseriesType(ds.attrs["timeseries_type"]),
                timeseries_region=TimeseriesRegion(ds.attrs["timeseries_region"]),
                measurement_name=ds.attrs["measurement_name"],
                measurements=ds["measurements"].to_series(),
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
        )

    preprocessor = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=AggregationPeriod(model.time_period),
        aggregation=AggregationStrategy(model.aggregation),
    )

    for output_path, xarray, ts in zip(output_paths, xarrays, preprocessor.timeseries):
        output = xr.Dataset(
            data_vars={
                "measurement": (["time"], ts.measurements.values),
                "std": (["time"], ts.measurement_std.values),
                "n": (["time"], ts.measurement_n.values),
                **(
                    {
                        "uncertainty": (["time"], ts.measurement_uncertainties.values),
                        "uncertainty_std": (["time"], ts.measurement_uncertainties_std.values),
                        "uncertainty_n": (["time"], ts.measurement_uncertainties_n.values),
                    }
                    if ts.measurement_uncertainties is not None
                    else {}
                ),
            },
            coords={
                "time": ts.measurements.index.values,
            },
            attrs=xarray.attrs,
        )
        output.to_zarr(output_path, mode="w")
