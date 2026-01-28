"""Entrypoint for timeseries module."""

import json
import warnings
from pathlib import Path

import cyclopts
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon, shape

from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    NormalisationType,
    TimeseriesType,
    UncertaintyType,
)

app = cyclopts.App()


@app.default
def main(
    label: str,
    dataset_path: Path,
    output_path: Path,
    measurement_name: str,
    measurement_uncertainty_name: str | None = None,
    mask_json_path: Path | None = None,
    apply_mask: bool = True,
    timeseries_type: TimeseriesType = TimeseriesType.CUMULATIVE,
    aggregate: AggregationStrategy = AggregationStrategy.MEAN,
    period: AggregationPeriod | None = AggregationPeriod.MONTH,
    uncertainty: UncertaintyType | None = None,
    independent: bool = False,
    t0_normalisation: NormalisationType | None = None,
) -> None:
    """
    Process a GeoZarr dataset into a timeseries.

    Parameters
    ----------
    label : str
        Label for the timeseries.
    dataset_path : Path
        Path to the input dataset .zarr file.
    output_path : Path
        Path to save the output .zarr file.
    measurement_name : str
        Name of the measurement to extract from the dataset.
    measurement_uncertainty_name : str | None, optional
        Name of the uncertainty to extract from the dataset.
    mask_json_path : Path | None, optional
        Path to the mask geometry GeoJSON file.
    apply_mask : bool, optional
        Whether the mask will be used to filter the data. Otherwise, the mask is only used to calculate coverage.
    timeseries_type : TimeseriesType, optional
        Type of timeseries to compute (cumulative or derivative).
    aggregate : AggregationStrategy, optional
        Aggregation strategy to apply when multiple measurements fall within the same time bin.
    period : AggregationPeriod | None, optional
        Time period for aggregation. If None, a single value will be computed for the entire timeseries.
    uncertainty : UncertaintyType | None, optional
        The statistics type describing the uncertainty.
    independent : bool, optional
        Whether the uncertainties are independent or not.
    t0_normalisation : NormalisationType | None, optional
        The type of normalisation to apply against time zero, if any.
    """
    timeseries = Timeseries(label=label, timeseries_type=timeseries_type)
    # Try opening the dataset - If for some reason we can't open it we propagate data=None forwards.
    try:
        data = xr.open_zarr(dataset_path)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Could not open dataset at {dataset_path} - {e}", stacklevel=1)
        data = None

    mask_geometry: Polygon | MultiPolygon | None
    if mask_json_path is not None:
        with mask_json_path.open("r") as f:
            geom = shape(json.load(f))
            assert isinstance(geom, (Polygon, MultiPolygon)), f"Invalid geometry {geom}"
            mask_geometry = geom
    else:
        mask_geometry = None

    timeseries.process_timeseries(
        data=data,
        measurement_name=measurement_name,
        measurement_uncertainty_name=measurement_uncertainty_name,
        mask_geometry=mask_geometry,
        apply_mask=apply_mask,
        aggregate=aggregate,
        period=period,
        uncertainty=uncertainty,
        independent=independent,
        t0_normalisation=str(t0_normalisation) if t0_normalisation is not None else None,
    )

    # if timeseries.measurements is not None:
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
                if measurement_uncertainty_name is not None
                else {}
            ),
        },
        coords={
            "time": timeseries.measurements.index.values,
        },
        attrs={
            "label": timeseries.label,
            "timeseries_type": timeseries.timeseries_type.value,
            "measurement_name": measurement_name,
            "measurement_uncertainty_name": measurement_uncertainty_name,
        },
    )

    output.to_zarr(output_path, mode="w")


if __name__ == "__main__":
    app()
