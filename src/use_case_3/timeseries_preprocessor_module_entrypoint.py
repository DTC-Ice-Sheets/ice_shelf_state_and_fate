"""Module entrypoint for the timeseries pre-processor module."""

import warnings
from pathlib import Path

import cyclopts
import xarray as xr

from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesType,
)

app = cyclopts.App()


@app.default
def main(
    input_paths: list[Path],
    output_path: Path,
    time_period: AggregationPeriod | None = None,
    aggregation: AggregationStrategy = AggregationStrategy.MEAN,
) -> None:
    """Pre-process a list of timeseries for covariate analysis.

    Parameters
    ----------
    input_paths : list[Path]
        List of paths to the input timeseries .zarr files
    output_path : Path
        Path to save the output .zarr file.
    time_period : AggregationPeriod | None, optional
        Time period for aggregation. If None, the lowest frequency from the input timeseries is used.
    aggregation : AggregationStrategy, optional
        Aggregation strategy to apply when multiple measurements fall within the same time bin.
    """
    timeseries = []
    xarrays = {}
    for path in input_paths:
        ds = xr.open_zarr(path)
        # Check if any of the input variables contain no data, as we don't want to include these from now on.
        if ds["measurement"] is not None:
            ts = Timeseries(
                label=ds.attrs["label"],
                timeseries_type=TimeseriesType(ds.attrs["timeseries_type"]),
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
        else:
            warnings.warn(f"No data containted in {path}, skipping this dataset", stacklevel=1)
            continue
        timeseries.append(ts)
        xarrays[ts.label] = ds

    preprocessor = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=time_period,
        aggregation=aggregation,
    )

    data_vars = {}
    timeseries_attrs = {}
    for label, ts in preprocessor.timeseries.items():
        key = f"{label}.{ts.measurement_name}"
        data_vars[key] = (["time"], ts.measurements.values)
        timeseries_attrs[key] = {
            "label": ts.label,
            "measurement_name": ts.measurement_name,
            "timeseries_type": ts.timeseries_type,
        }

        if ts.measurement_uncertainties is not None:
            uncertainty_key = f"{label}.{xarrays[label].attrs['measurement_uncertainty_name']}"
            data_vars[uncertainty_key] = (["time"], ts.measurement_uncertainties.values)
            timeseries_attrs[key]["uncertainty"] = uncertainty_key

    output = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": next(iter(preprocessor.timeseries.values())).measurements.index.values,
        },
        attrs={
            "time_period": preprocessor.aggregation_period,
            "aggregation": preprocessor.aggregation,
            "timeseries": timeseries_attrs,
        },
    )
    output.to_zarr(output_path, mode="w")


if __name__ == "__main__":
    app()
