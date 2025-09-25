"""Entrypoint for the covariate analyser module."""

import argparse
from types import UnionType
from typing import get_args

import xarray as xr

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.schemas import CovariateAnalyserModuleEntryPointSchema
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesRegion,
    TimeseriesType,
)


def main() -> None:
    """Execute the covariate analyser module from CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="use_case_3.covariate_analyser_module",
        description="Plot covariate analysis from a set of pre-processed timeseries.",
    )

    # Add arguments to the parser
    for name, field in CovariateAnalyserModuleEntryPointSchema.model_fields.items():
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
            required=name in ["input_path"],
        )

    # Parse the input arguments
    args = parser.parse_args()
    model = CovariateAnalyserModuleEntryPointSchema(**vars(args))

    timeseries = []
    ds = xr.open_zarr(model.input_path)
    for name, attr in ds.attrs["timeseries"].items():
        label, measurement_name = name.split(".", 1)
        timeseries.append(
            Timeseries(
                label=label,
                timeseries_type=TimeseriesType(attr["timeseries_type"]),
                timeseries_region=TimeseriesRegion(attr["timeseries_region"]),
                measurement_name=measurement_name,
                measurements=ds[name].to_series(),
                **(
                    {
                        "measurement_uncertainties": ds[attr["uncertainty_name"]].to_series(),
                    }
                    if "uncertainty_name" in attr
                    else {}
                ),
            )
        )

    preprocessor = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=AggregationPeriod(ds.attrs["time_period"]),
        aggregation=AggregationStrategy(ds.attrs["aggregation"]),
        process_timeseries=False,
    )
    analyser = CovariateAnalyser(preprocessor)

    fig, _ = analyser.jointplot_grid()

    fig.savefig(model.output_path)


if __name__ == "__main__":
    main()
