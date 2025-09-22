"""Entrypoint for the covariate analyser module."""

import argparse
import logging
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

logger = logging.getLogger(__name__)


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
    for name, var in ds.data_vars.items():
        if name.endswith("_uncertainty"):
            continue
        timeseries.append(
            Timeseries(
                label=name,
                timeseries_type=TimeseriesType.CUMULATIVE,
                timeseries_region=TimeseriesRegion.ICE_SHELF,
                measurement_name=name,
                measurements=var.to_series(),
            )
        )

    preprocessor = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )
    analyser = CovariateAnalyser(preprocessor)

    fig, _ = analyser.plot_timeseries()

    fig.savefig(model.output_path)


if __name__ == "__main__":
    main()
