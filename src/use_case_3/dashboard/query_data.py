import sys

import geopandas as gpd
from shapely import MultiPolygon

module_path = "/home/palmerjohn/code/dtc_ice_sheets/data_download_module/src/"
sys.path.insert(0, module_path)
module_path = "/home/palmerjohn/code/dtc_ice_sheets/data_converter/src"
sys.path.insert(0, module_path)

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from data_download_module.main import download_data
from data_download_module.schemas import DataDownloadRequest

import use_case_3.helper_functions as hf
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesRegion,
    TimeseriesType,
    UncertaintyType,
)

OUTPUT_DIRECTORY = Path("/data/yak1/working/dtc_ice_sheets/dashboard_download_scratch")

with open("/home/palmerjohn/code/dtc_ice_sheets/use_case_3/src/use_case_3/dashboard/variable_mappings.yml") as f:
    config_dict = yaml.safe_load(f)


def query_ice_shelf_wide_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    local_raw_request = DataDownloadRequest(dataset_id="ICE_SHELF_MASKS", raw_download=True)

    downloaded_parquet_data_path = download_data(
        data_download_request=local_raw_request, output_parent_directory=OUTPUT_DIRECTORY
    )

    ice_shelf_polygons_proj = gpd.read_parquet(downloaded_parquet_data_path)
    ice_shelf_polygons_proj = ice_shelf_polygons_proj.sort_values(["Name"])
    ice_shelf_polygons_epsg4326 = ice_shelf_polygons_proj.to_crs("EPSG:4326")

    return ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj


def query_all_ocean_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    local_raw_request = DataDownloadRequest(dataset_id="OCEAN_MASKS", raw_download=True)

    downloaded_parquet_data_path = download_data(
        data_download_request=local_raw_request, output_parent_directory=OUTPUT_DIRECTORY
    )

    ocean_polygons_proj = gpd.read_parquet(downloaded_parquet_data_path)
    ocean_polygons_proj = ocean_polygons_proj.sort_values(["Name"])
    ocean_polygons_epsg4326 = ocean_polygons_proj.to_crs("EPSG:4326")

    return ocean_polygons_epsg4326, ocean_polygons_proj


def query_polygons() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    local_raw_request = DataDownloadRequest(dataset_id="POLYGONS", raw_download=True)

    downloaded_parquet_data_path = download_data(
        data_download_request=local_raw_request, output_parent_directory=OUTPUT_DIRECTORY
    )

    polygons_epsg4326 = gpd.read_parquet(downloaded_parquet_data_path)
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

    ice_shelf_polygons_proj = ice_shelf_polygons_epsg4326.to_crs("EPSG:3031")
    ocean_polygons_proj = ocean_polygons_epsg4326.to_crs("EPSG:3031")

    return ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj, ocean_polygons_epsg4326, ocean_polygons_proj


def query_ice_sheet_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    local_raw_request = DataDownloadRequest(dataset_id="ICE_SHEET_MASKS", raw_download=True)

    downloaded_parquet_data_path = download_data(
        data_download_request=local_raw_request, output_parent_directory=OUTPUT_DIRECTORY
    )

    ice_sheet_polygons_proj = gpd.read_parquet(downloaded_parquet_data_path)
    ice_sheet_polygons_proj = ice_sheet_polygons_proj.sort_values(["Name"])
    ice_sheet_polygons_epsg4326 = ice_sheet_polygons_proj.to_crs("EPSG:4326")

    return ice_sheet_polygons_epsg4326, ice_sheet_polygons_proj


def query_summary_data(
    ice_shelf_polygons: gpd.GeoDataFrame,
    ice_shelf: str = "Thwaites",
    year_range: list[pd.Timestamp | np.datetime64 | None, pd.Timestamp | np.datetime64 | None] | None = None,
) -> tuple[xr.Dataset, MultiPolygon, MultiPolygon]:
    if year_range is None:
        year_range = [None, None]

    ice_shelf_polygon_epsg4326, ice_shelf_polygon = hf.iceshelf_polygon(
        ice_shelf_polygons=ice_shelf_polygons,
        ice_shelf_name=ice_shelf,
    )

    polygons = gpd.GeoDataFrame({"Name": ["ice_shelf", "ocean", "climate"], "geometry": [None, None, None]})
    polygons = polygons.set_crs("epsg: 4326").set_index("Name")

    local_raw_request = DataDownloadRequest(dataset_id=f"{ice_shelf}_SUMMARY".upper(), raw_download=False)

    summary_data_path = download_data(data_download_request=local_raw_request, output_parent_directory=OUTPUT_DIRECTORY)

    summary_data = xr.open_zarr(store=summary_data_path, zarr_format=2)

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

    if year_range is None:
        start_date, end_date = timeseries_container._find_start_end_dates()
    else:
        if year_range[0] is None and year_range[1] is None:
            start_date, end_date = timeseries_container._find_start_end_dates()
        elif year_range[0] is None and year_range[1] is not None:
            start_date, _ = timeseries_container._find_start_end_dates()
            end_date = year_range[1]
        elif year_range[0] is None and year_range[1] is not None:
            _, end_date = timeseries_container._find_start_end_dates()
            start_date = year_range[0]
        else:
            start_date = year_range[0]
            end_date = year_range[1]

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

    print(dataset_download_ids)

    timeseries_list = []
    for dataset_download_id in dataset_download_ids:
        local_request = DataDownloadRequest(
            dataset_id=dataset_download_id,
            start_time=start_time,
            end_time=end_time,
            epsg4326_polygon=polygons.loc[region, "geometry"],
            raw_download=False,
        )

        download_path = download_data(data_download_request=local_request, output_parent_directory=OUTPUT_DIRECTORY)

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
            timeseries_list.append(timeseries)

    return timeseries_list


def get_all_ocean_moorings():
    return [
        "OCEAN_MOORINGS_" + mooring.split(".")[0].upper()
        for mooring in os.listdir("/data/yak1/working/dtc_ice_sheets/initial_datasets/ocean_moorings/")
    ]


def get_all_sg_lakes():
    return [
        "SUGLACIAL_LAKE_" + sg_lake.split(".")[0].upper()
        for sg_lake in os.listdir("/data/yak1/working/dtc_ice_sheets/initial_datasets/subglacial_lakes/")
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
