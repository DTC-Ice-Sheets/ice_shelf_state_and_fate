"""Functions for querying data from the dtc_query_api.

Some functions are not yet using the dtc_query_api, but they should be migrated eventually.
"""

import functools
import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import boto3
import geopandas as gpd
import xarray as xr
import yaml
from data_download_module.main import download_data
from data_download_module.schemas import DataDownloadRequest
from dtc_query_client import ApiClient, Configuration, IceShelvesApi
from pyproj import CRS
from shapely import MultiPolygon, Polygon

from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesType,
    UncertaintyType,
)

api_config = Configuration(host="http://localhost:8000")


url = "http://localhost:8000"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    endpoint_url=os.environ["AWS_S3_ENDPOINT"],
    region_name=os.environ["AWS_DEFAULT_REGION"],
    use_ssl=True,
)

output_path = Path("/data/yak1/working/dtc_ice_sheets/dashboard_download_scratch")
polygon_path = Path("src/use_case_3/dashboard/assets/polygons.parquet")
icesheet_polygon_path = Path("src/use_case_3/dashboard/assets/icesheet_polygon.parquet")
initial_datasets_path = Path("/data/yak1/working/dtc_ice_sheets/initial_datasets")
config_dict_path = Path("src/use_case_3/dashboard/variable_mappings.yml")

# Load metadata about the available datasets and variables
with config_dict_path.open("r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)


class ProjectedGeoDataFrame:
    """
    Helper class to store multiple projections of the same GeoDataFrame in a lazily cached way.

    Parameters
    ----------
    epsg4326 : gpd.GeoDataFrame | None
        The GeoDataFrame in EPSG:4326 projection.
        Must be provided if `proj` is not provided.
    proj : gpd.GeoDataFrame | None
        The GeoDataFrame in the projection specified by `crs`.
        Must be provided if `epsg4326` is not provided.
        Must be provided if `crs` is not provided.
    crs : str | CRS | None
        The coordinate reference system of the `proj` GeoDataFrame.
        Must be provided if `proj` is not provided.
    """

    def __init__(
        self,
        epsg4326: gpd.GeoDataFrame | None = None,
        proj: gpd.GeoDataFrame | None = None,
        crs: str | CRS | None = None,
    ) -> None:
        assert (epsg4326 is not None) or (proj is not None), "At least one of epsg4326 or proj must be provided"
        assert (proj is not None) or (crs is not None), "If proj is not provided, a CRS must be specified"
        self._epsg4326 = epsg4326
        self._proj = proj
        self.crs = CRS(crs or proj.crs)

    @property
    def epsg4326(self) -> gpd.GeoDataFrame:
        """The GeoDataFrame in EPSG:4326 projection."""
        if self._epsg4326 is None:
            self._epsg4326 = self._proj.to_crs("EPSG:4326")
        return self._epsg4326

    @property
    def proj(self) -> gpd.GeoDataFrame:
        """The GeoDataFrame in the projection specified by `crs`."""
        if self._proj is None:
            self._proj = self._epsg4326.to_crs("EPSG:3031")
        return self._proj


class RegionPolygons(TypedDict):
    """A set of available polygons for the data masking of an ice shelf."""

    ice_shelf: Polygon | MultiPolygon | None
    ocean: Polygon | MultiPolygon | None
    climate: Polygon | MultiPolygon | None


@functools.cache
def query_polygons() -> tuple[ProjectedGeoDataFrame, ProjectedGeoDataFrame]:
    """Get the polygon geometries for all ice shelves.

    Returns
    -------
    tuple[ProjectedGeoDataFrame, ProjectedGeoDataFrame]
        A tuple containing:
        - A polygon of the ice shelf
        - A polygon of the surrounding ocean
    """
    polygons_epsg4326 = gpd.read_parquet(polygon_path)
    polygons_epsg4326 = polygons_epsg4326.sort_values(["Name"])
    ice_shelf_polygons_epsg4326 = (
        polygons_epsg4326.copy()
        .drop(["geometry_ocean"], axis=1)
        .rename({"geometry_ice_shelf": "geometry"}, axis=1)
        .set_geometry("geometry")
    )
    ocean_polygons_epsg4326 = (
        polygons_epsg4326.copy()
        .drop(["geometry_ice_shelf"], axis=1)
        .rename({"geometry_ocean": "geometry"}, axis=1)
        .set_geometry("geometry")
    )

    return (
        ProjectedGeoDataFrame(epsg4326=ice_shelf_polygons_epsg4326, crs="EPSG:3031"),
        ProjectedGeoDataFrame(epsg4326=ocean_polygons_epsg4326, crs="EPSG:3031"),
    )


@functools.cache
def get_ice_shelf_polygons(ice_shelf: str) -> RegionPolygons:
    """Get the polygon geometries for a given ice shelf."""
    ice_shelf_polygons, ocean_polygons = query_polygons()
    ice_shelf_extent = ice_shelf_polygons.epsg4326[ice_shelf_polygons.epsg4326["Name"] == ice_shelf]["geometry"].iloc[0]
    ice_shelf_extent = ice_shelf_extent.buffer(10)
    ocean_extent = ocean_polygons.epsg4326[ocean_polygons.epsg4326["Name"] == ice_shelf]["geometry"].iloc[0]
    climate_extent = None

    return RegionPolygons(
        ice_shelf=ice_shelf_extent,
        ocean=ocean_extent,
        climate=climate_extent,
    )


@functools.cache
def query_ice_sheet_data() -> ProjectedGeoDataFrame:
    """Get the polygon geometry for the entire ice sheet."""
    ice_sheet_polygons_proj = gpd.read_parquet(icesheet_polygon_path)
    ice_sheet_polygons_proj = ice_sheet_polygons_proj.sort_values(["Name"])
    return ProjectedGeoDataFrame(proj=ice_sheet_polygons_proj)


async def query_summary_data(
    ice_shelf: str = "thwaites",
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[xr.Dataset, TimeSeriesPreprocessor]:
    """Get the summary data for a given ice shelf."""
    async with ApiClient(api_config) as client:
        summary_zarr_url = (await IceShelvesApi(client).get_summary_url(ice_shelf_id=ice_shelf)).url

    summary_data = xr.open_dataset(
        summary_zarr_url,
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

    timeseries_list = []

    for var in summary_data:
        if var in config_dict["timeseries_summary"]:
            timeseries = Timeseries(
                label=var,
                timeseries_type=TimeseriesType(config_dict["timeseries_summary"][var]["timeseries_type"]),
                unit=config_dict["timeseries_summary"][var]["units"],
            )

            timeseries.process_timeseries(
                data=summary_data,
                measurement_name=var,
                measurement_uncertainty_name=config_dict["timeseries_summary"][var]["measurement_uncertainties"],
                mask_geometry=None,
                apply_mask=False,
                aggregate=AggregationStrategy.MEAN,
                period=AggregationPeriod.YEAR,
                uncertainty=UncertaintyType(config_dict["timeseries_summary"][var]["measurement_uncertainty_type"]),
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
) -> list[Timeseries]:
    if dataset_id == "OCEAN_MOORINGS":
        dataset_download_ids = _get_all_ocean_moorings()
    elif dataset_id == "SUB_GLACIAL_LAKES":
        dataset_download_ids = _get_all_sg_lakes()
    else:
        dataset_download_ids = [dataset_id]

    timeseries_list = []
    for dataset_download_id in dataset_download_ids:
        local_request = DataDownloadRequest(
            dataset_id=dataset_download_id,
            start_time=start_time,
            end_time=end_time,
            epsg4326_polygon=mask_geometry,
            raw_download=False,
            output_parent_path=output_path,
        )

        download_path = download_data(data_download_request=local_request)

        data = xr.open_dataset(download_path, engine="zarr")

        data = data.rio.write_crs(data.spatial_ref.spatial_ref)

        include_data = True
        for dataset_var in data:
            if xr.ufuncs.isnan(data[dataset_var]).all():
                include_data = False

        if include_data:
            timeseries = Timeseries(
                label=dataset_download_id + "-" + variable_id,
                timeseries_type=TimeseriesType(
                    config_dict[dataset_id.lower()]["variables"][variable_id]["timeseries_type"]
                ),
                unit=config_dict[dataset_id.lower()]["variables"][variable_id]["units"],
            )
            timeseries.process_timeseries(
                data=data,
                measurement_name=variable_id,
                measurement_uncertainty_name=config_dict[dataset_id.lower()]["variables"][variable_id][
                    "measurement_uncertainties"
                ],
                mask_geometry=mask_geometry,
                apply_mask=False,
                aggregate=AggregationStrategy.MEAN,
                period=AggregationPeriod.MONTH,
                uncertainty=UncertaintyType(
                    config_dict[dataset_id.lower()]["variables"][variable_id]["measurement_uncertainty_type"]
                ),
                independent=True,
                t0_normalisation=None,
            )
    return timeseries_list


def _get_all_ocean_moorings() -> list[str]:
    return [
        "OCEAN_MOORINGS_" + mooring.with_suffix("").name.upper()
        for mooring in (initial_datasets_path / "ocean_moorings").iterdir()
    ]


def _get_all_sg_lakes() -> list[str]:
    return [
        "SUGLACIAL_LAKE_" + sg_lake.with_suffix("").name.upper()
        for sg_lake in (initial_datasets_path / "subglacial_lakes").iterdir()
    ]


def process_all_data(
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
    polygons = get_ice_shelf_polygons(ice_shelf)

    timeseries_list = []
    for dataset_id, variable_id in variables:
        timeseries_region = config_dict[dataset_id.lower()]["variables"][variable_id]["region"]
        mask_geometry = polygons[timeseries_region]

        timeseries = _query_local_data(
            dataset_id=dataset_id.upper(),
            variable_id=variable_id,
            mask_geometry=mask_geometry,
            start_time=start_date,
            end_time=end_date,
        )
        for ts in timeseries:
            timeseries_list.append(ts)
    return timeseries_list
