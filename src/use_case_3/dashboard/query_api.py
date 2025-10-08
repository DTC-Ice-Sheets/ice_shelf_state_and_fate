import functools
import os
import sys
from datetime import datetime
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml
from pyproj import CRS
from shapely import MultiPolygon

from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.helper_functions import iceshelf_polygon
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesRegion,
    TimeseriesType,
    UncertaintyType,
)

module_path = "/home/palmerjohn/code/dtc_ice_sheets/data_download_module/src/"
sys.path.insert(0, module_path)
module_path = "/home/palmerjohn/code/dtc_ice_sheets/data_converter/src"
sys.path.insert(0, module_path)

from data_download_module.main import download_data
from data_download_module.schemas import DataDownloadRequest

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


with config_dict_path.open() as f:
    config_dict = yaml.safe_load(f)


class ProjectedGeoDataFrame:
    def __init__(
        self,
        epsg4326: gpd.GeoDataFrame | None = None,
        proj: gpd.GeoDataFrame | None = None,
        crs: str | CRS | None = None,
    ):
        assert (epsg4326 is not None) or (proj is not None), "At least one of epsg4326 or proj must be provided"
        assert (proj is not None) or (crs is not None), "If proj is not provided, a CRS must be specified"
        self._epsg4326 = epsg4326
        self._proj = proj
        self.crs = CRS(crs or proj.crs)

    @property
    def epsg4326(self) -> gpd.GeoDataFrame:
        if self._epsg4326 is None:
            self._epsg4326 = self._proj.to_crs("EPSG:4326")
        return self._epsg4326

    @property
    def proj(self) -> gpd.GeoDataFrame:
        if self._proj is None:
            self._proj = self._epsg4326.to_crs("EPSG:3031")
        return self._proj


@functools.cache
def query_ice_shelf_wide_data() -> ProjectedGeoDataFrame:
    local_raw_request = DataDownloadRequest(
        dataset_id="ICE_SHELF_MASKS", raw_download=True, output_parent_path=output_path
    )

    downloaded_parquet_data_path = download_data(data_download_request=local_raw_request)

    ice_shelf_polygons_proj = gpd.read_parquet(downloaded_parquet_data_path)
    ice_shelf_polygons_proj = ice_shelf_polygons_proj.sort_values(["Name"])

    return ProjectedGeoDataFrame(proj=ice_shelf_polygons_proj)


@functools.cache
def query_all_ocean_data() -> ProjectedGeoDataFrame:
    local_raw_request = DataDownloadRequest(dataset_id="OCEAN_MASKS", raw_download=True, output_parent_path=output_path)

    downloaded_parquet_data_path = download_data(data_download_request=local_raw_request)

    ocean_polygons_proj = gpd.read_parquet(downloaded_parquet_data_path)
    ocean_polygons_proj = ocean_polygons_proj.sort_values(["Name"])

    return ProjectedGeoDataFrame(proj=ocean_polygons_proj)


@functools.cache
def query_polygons() -> tuple[ProjectedGeoDataFrame, ProjectedGeoDataFrame]:
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
def query_ice_sheet_data() -> ProjectedGeoDataFrame:
    ice_sheet_polygons_proj = gpd.read_parquet(icesheet_polygon_path)
    ice_sheet_polygons_proj = ice_sheet_polygons_proj.sort_values(["Name"])
    return ProjectedGeoDataFrame(proj=ice_sheet_polygons_proj)


@functools.cache
def query_summary_data(
    ice_shelf: str = "thwaites",
    start_date: pd.Timestamp | np.datetime64 | None = None,
    end_date: pd.Timestamp | np.datetime64 | None = None,
) -> tuple[xr.Dataset, TimeSeriesPreprocessor, MultiPolygon, MultiPolygon]:
    ice_shelf_polygon_epsg4326, ice_shelf_polygon = iceshelf_polygon(
        ice_shelf_polygons=query_polygons()[0].proj,
        ice_shelf_name=ice_shelf,
    )

    polygons = gpd.GeoDataFrame({"Name": ["ice_shelf", "ocean", "climate"], "geometry": [None, None, None]})
    polygons = polygons.set_crs("epsg: 4326").set_index("Name")

    response = requests.get(f"{url}/ice-shelves/{ice_shelf}/summary-url")
    summary_zarr_url = response.json()["url"]

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

    for var in summary_data.keys():
        if var in config_dict["timeseries_summary"]:
            timeseries = Timeseries(
                label=var,
                timeseries_type=TimeseriesType(config_dict["timeseries_summary"][var]["timeseries_type"]),
                timeseries_region=TimeseriesRegion(config_dict["timeseries_summary"][var]["region"]),
                unit=config_dict["timeseries_summary"][var]["units"],
            )

            timeseries.process_timeseries(
                data=summary_data,
                measurement_name=var,
                measurement_uncertainty_name=config_dict["timeseries_summary"][var]["measurement_uncertainties"],
                mask_geometries=polygons,
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

    if start_date is None or end_date is None:
        min_date, max_date = timeseries_container._find_start_end_dates()
        start_date = start_date or min_date
        end_date = end_date or max_date

    timeseries_container._trim_timeseries_date(start_date=start_date, end_date=end_date)

    return summary_data, timeseries_container, ice_shelf_polygon_epsg4326, ice_shelf_polygon


def query_local_data(dataset_id, variable, polygons, start_time, end_time):
    region = config_dict[dataset_id.lower()]["variables"][variable]["region"]
    if dataset_id == "OCEAN_MOORINGS":
        dataset_download_ids = get_all_ocean_moorings()
    elif dataset_id == "SUB_GLACIAL_LAKES":
        dataset_download_ids = get_all_sg_lakes()
    else:
        dataset_download_ids = [dataset_id]

    timeseries_list = []
    for dataset_download_id in dataset_download_ids:
        local_request = DataDownloadRequest(
            dataset_id=dataset_download_id,
            start_time=start_time,
            end_time=end_time,
            epsg4326_polygon=polygons.loc[region, "geometry"],
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
                label=dataset_download_id + "-" + variable,
                timeseries_type=TimeseriesType(
                    config_dict[dataset_id.lower()]["variables"][variable]["timeseries_type"]
                ),
                timeseries_region=TimeseriesRegion(config_dict[dataset_id.lower()]["variables"][variable]["region"]),
                unit=config_dict[dataset_id.lower()]["variables"][variable]["units"],
            )
            timeseries.process_timeseries(
                data=data,
                measurement_name=variable,
                measurement_uncertainty_name=config_dict[dataset_id.lower()]["variables"][variable][
                    "measurement_uncertainties"
                ],
                mask_geometries=polygons,
                apply_mask=False,
                aggregate=AggregationStrategy.MEAN,
                period=AggregationPeriod.MONTH,
                uncertainty=UncertaintyType(
                    config_dict[dataset_id.lower()]["variables"][variable]["measurement_uncertainty_type"]
                ),
                independent=True,
                t0_normalisation=None,
            )
    return timeseries_list


def get_all_ocean_moorings():
    return [
        "OCEAN_MOORINGS_" + mooring.with_suffix("").name.upper()
        for mooring in (initial_datasets_path / "ocean_moorings").iterdir()
    ]


def get_all_sg_lakes():
    return [
        "SUGLACIAL_LAKE_" + sg_lake.with_suffix("").name.upper()
        for sg_lake in (initial_datasets_path / "subglacial_lakes").iterdir()
    ]


def process_all_data(variables, polygons, year_range):
    timeseries_list = []
    for i, var in enumerate(variables):
        # if variable_states[i]:
        dataset_id = var.split("-")[0]
        var_id = var.split("-")[-1]

        timeseries = query_local_data(
            dataset_id=dataset_id.upper(),
            variable=var_id,
            polygons=polygons,
            start_time=datetime(year_range[0], 1, 1),
            end_time=datetime(year_range[1], 1, 1),
        )
        for ts in timeseries:
            timeseries_list.append(ts)

    return timeseries_list
