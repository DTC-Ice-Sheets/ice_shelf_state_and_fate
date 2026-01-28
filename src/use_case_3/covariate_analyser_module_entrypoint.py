"""Entrypoint for the covariate analyser module."""

import json
import warnings
from pathlib import Path

import cyclopts
import xarray as xr

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    JointPlotAnalysisType,
    TimeseriesType,
)

app = cyclopts.App()


@app.default
def main(
    input_path: Path,
    output_path: Path,
    analysis_type: JointPlotAnalysisType = JointPlotAnalysisType.CORR,
) -> None:
    """Plot covariate analysis from a set of pre-processed timeseries.

    Parameters
    ----------
    input_path : Path
        Path to the input .zarr file containing pre-processed timeseries.
    output_path : Path
        Path to save the output .json file.
    analysis_type : JointPlotAnalysisType, optional
        Type of covariate analysis to perform.
    """
    timeseries = []
    ds = xr.open_zarr(input_path)
    for key, attr in ds.attrs["timeseries"].items():
        if ds[key] is not None:
            label = attr["label"]
            measurement_name = attr["measurement_name"]
            timeseries.append(
                Timeseries(
                    label=label,
                    timeseries_type=TimeseriesType(attr["timeseries_type"]),
                    measurement_name=measurement_name,
                    measurements=ds[key].to_series(),
                    **(
                        {
                            "measurement_uncertainties": ds[attr["uncertainty"]].to_series(),
                        }
                        if "uncertainty" in attr
                        else {}
                    ),
                )
            )
        else:
            warnings.warn(f"No data containted in {measurement_name}, skipping this variable", stacklevel=1)
            continue

    if len(timeseries) < 2:
        raise ValueError(
            f"Only {len(timeseries)} valid timeseries found in the input dataset for covariate analysis. "
            "At least 2 are required."
        )
    preprocessor = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=AggregationPeriod(ds.attrs["time_period"]),
        aggregation=AggregationStrategy(ds.attrs["aggregation"]),
        process_timeseries=False,
    )
    analyser = CovariateAnalyser(preprocessor)

    results = analyser.run_analysis(
        analysis_type=analysis_type,
    )

    with output_path.open("w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    app()
