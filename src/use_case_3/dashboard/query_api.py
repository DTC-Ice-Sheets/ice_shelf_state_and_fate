"""Helper functions for working with the DTC Query API.

Mostly these are simple wrappers that provide memoization for common API queries.

Some functions are not yet using the dtc_query_api, but they should be migrated eventually.
"""

import os
from datetime import datetime
from pathlib import Path

import asyncstdlib
import boto3
import xarray as xr
import yaml
from data_download_module.main import download_data
from data_download_module.schemas import DataDownloadRequest
from dtc_query_client import ApiClient, Configuration, IceShelfExtent, StateAndFateApi
from shapely import MultiPolygon, Polygon
from shapely.geometry import shape

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

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    endpoint_url=os.environ["AWS_S3_ENDPOINT"],
    region_name=os.environ["AWS_DEFAULT_REGION"],
    use_ssl=True,
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


@asyncstdlib.cache
async def get_ice_shelves() -> list[str]:
    """List the available ice shelves."""
    async with ApiClient(api_config) as client:
        return await StateAndFateApi(client).list_ice_shelves()


@asyncstdlib.lru_cache(128)
async def get_ice_shelf_geometry(ice_shelf: str, extent: IceShelfExtent) -> MultiPolygon:
    """Query and parse the geometry for a given ice shelf and extent type."""
    async with ApiClient(api_config) as client:
        return shape((await StateAndFateApi(client).get_geometry(ice_shelf, extent)).to_dict())


@asyncstdlib.cache
async def _get_summary_zarr_url(ice_shelf: str) -> str:
    async with ApiClient(api_config) as client:
        return (await StateAndFateApi(client).get_summary_url(ice_shelf_id=ice_shelf)).url


def _open_s3_zarr(s3_url: str) -> xr.Dataset:
    return xr.open_dataset(
        s3_url,
        engine="zarr",
        backend_kwargs={
            "storage_options": {
                "key": os.environ["AWS_ACCESS_KEY_ID"],
                "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
                "client_kwargs": {"endpoint_url": os.environ["AWS_S3_ENDPOINT"]},
            }
        },
        chunks={},
    )


@asyncstdlib.cache
async def get_ice_shelf_time_range(ice_shelf: str) -> tuple[datetime, datetime]:
    """Get the time range of the summary data for a given ice shelf."""
    summary_data = _open_s3_zarr(await _get_summary_zarr_url(ice_shelf=ice_shelf))
    min_time = summary_data["time"].min().item()
    max_time = summary_data["time"].max().item()
    return min_time, max_time


###################################################################
# Functions below this line should be migrated to use the DTC API #
###################################################################


@asyncstdlib.lru_cache(128)
async def query_summary_data(
    ice_shelf: str = "thwaites",
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[xr.Dataset, TimeSeriesPreprocessor]:
    """Get the summary data for a given ice shelf."""
    summary_data = _open_s3_zarr(await _get_summary_zarr_url(ice_shelf=ice_shelf))
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

    if start_year is None or end_year is None:
        min_year, max_year = timeseries_container.find_start_end_dates()
        start_year = start_year or min_year
        end_year = end_year or max_year

    timeseries_container.trim_start_end_dates(start_date=start_year, end_date=end_year)

    return summary_data, timeseries_container


def _query_local_data(
    dataset_id: str,
    variable_id: str,
    mask_geometry: Polygon | MultiPolygon | None,
    start_time: datetime,
    end_time: datetime,
) -> Timeseries:
    local_request = DataDownloadRequest(
        dataset_id=dataset_id,
        start_time=start_time,
        end_time=end_time,
        epsg4326_polygon=mask_geometry,
        raw_download=False,
        output_parent_path=output_path,
    )

    download_path = download_data(data_download_request=local_request)

    data = xr.open_dataset(download_path, engine="zarr")

    data = data.rio.write_crs(data.spatial_ref.spatial_ref)

    timeseries = Timeseries(
        label=dataset_id + "-" + variable_id,
        timeseries_type=TimeseriesType(config_dict_local[dataset_id]["variables"][variable_id]["timeseries_type"]),
        unit=config_dict_local[dataset_id]["variables"][variable_id]["units"],
    )

    uncertainty_type = config_dict_local[dataset_id]["variables"][variable_id]["measurement_uncertainty_type"]
    if uncertainty_type is not None:
        uncertainty_type = UncertaintyType(uncertainty_type)

    timeseries.process_timeseries(
        data=data,
        measurement_name=variable_id,
        measurement_uncertainty_name=config_dict_local[dataset_id]["variables"][variable_id][
            "measurement_uncertainties"
        ],
        mask_geometry=mask_geometry,
        apply_mask=False,
        aggregate=AggregationStrategy.MEAN,
        period=AggregationPeriod.MONTH,
        uncertainty=uncertainty_type,
        independent=True,
        t0_normalisation=None,
    )
    return timeseries


async def process_all_data(
    ice_shelf: str,
    variables: list[tuple[str, str]],
    start_date: datetime,
    end_date: datetime,
) -> list[Timeseries]:
    """
    Download all required data and process it locally into Timeseries objects.

    Parameters
    ----------
    ice_shelf : str
        The name of the ice shelf to query data for.
    variables : list[tuple[str, str]]
        A list of tuples: (dataset_id, variable_id)
    start_date : datetime
        The start date for the time series.
    end_date : datetime
        The end date for the time series.

    Returns
    -------
    list[Timeseries]
        A list of processed Timeseries objects.
    """
    timeseries_list = []
    for dataset_id, variable_id in variables:
        timeseries_region = config_dict_local[dataset_id]["region"]
        mask_geometry = await get_ice_shelf_geometry(ice_shelf, IceShelfExtent(timeseries_region))

        timeseries = _query_local_data(
            dataset_id=dataset_id,
            variable_id=variable_id,
            mask_geometry=mask_geometry,
            start_time=start_date,
            end_time=end_date,
        )
        timeseries_list.append(timeseries)
    return timeseries_list
