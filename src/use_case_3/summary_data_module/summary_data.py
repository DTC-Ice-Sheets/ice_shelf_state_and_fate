"""Module to calculate simple summary data from time series."""

import os
from pathlib import Path

import xarray as xr
import yaml
from dtc_query_client import ApiClient, Configuration, StateAndFateApi

from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesType,
    UncertaintyType,
)

api_config = Configuration(
    host=os.environ.get("DTC_API_URL", "https://query.dtc-ice-sheets.org"),
    access_token=os.environ["DTC_API_KEY"],
)

output_path = Path("/data/yak1/working/dtc_ice_sheets/dashboard_download_scratch")
initial_datasets_path = Path("/data/yak1/working/dtc_ice_sheets/initial_datasets")
config_dict_path = Path("src/use_case_3/dashboard/variable_mappings.yml")
config_dict_local_path = Path("src/use_case_3/dashboard/variable_mappings_local.yml")

# Load metadata about the available datasets and variables
with config_dict_path.open("r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)
with config_dict_local_path.open("r", encoding="utf-8") as f:
    config_dict_local = config_dict.copy() | yaml.safe_load(f)


async def timeseries_zarr_to_json(ice_shelf: str) -> dict:
    """Get the summary data for a given ice shelf from a Zarr file."""
    summary_data = _open_zarr(await _get_summary_zarr_url(ice_shelf))
    metadata = config_dict_local["timeseries_summary"]

    timeseries_list = []
    for var in summary_data:
        if var in metadata:
            timeseries = Timeseries(
                label=var,
                timeseries_type=TimeseriesType(metadata[var]["timeseries_type"]),
                unit=metadata[var]["units"],
            )

            timeseries.process_timeseries(
                data=summary_data,
                measurement_name=var,
                measurement_uncertainty_name=metadata[var]["measurement_uncertainties"],
                mask_geometry=None,
                apply_mask=False,
                aggregate=AggregationStrategy.MEAN,
                period=AggregationPeriod.YEAR,
                uncertainty=UncertaintyType(metadata[var]["measurement_uncertainty_type"]),
                independent=True,
                t0_normalisation=None,
            )

            timeseries_list.append(timeseries)

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=timeseries_list,
        time_period=AggregationPeriod.YEAR,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    start_year, end_year = timeseries_container.find_start_end_dates()

    timeseries_container.trim_start_end_dates(start_date=start_year, end_date=end_year)

    plot_dict = {}
    for timeseries_label in timeseries_container.timeseries:
        timeseries = timeseries_container.timeseries[timeseries_label]
        plot_dict[timeseries_label] = {
            "time": timeseries.measurements.index.tolist(),
            "measurements": timeseries.measurements.tolist(),
            "uncertainties": timeseries.measurement_uncertainties.tolist()
            if timeseries.measurement_uncertainties is not None
            else None,
        }

    return plot_dict


def _open_zarr(url: str) -> xr.Dataset:
    is_s3_url = url.lower().startswith("s3://")
    return xr.open_dataset(
        url,
        engine="zarr",
        backend_kwargs=(
            {
                "storage_options": {
                    "key": os.environ["AWS_ACCESS_KEY_ID"],
                    "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
                    "client_kwargs": {"endpoint_url": os.environ["AWS_S3_ENDPOINT"]},
                }
            }
            if is_s3_url
            else {}
        ),
        chunks={},
    )


async def _get_summary_zarr_url(ice_shelf: str) -> str:
    async with ApiClient(api_config) as client:
        return (await StateAndFateApi(client).get_summary_url(ice_shelf_id=ice_shelf)).url
