import sys

import geopandas as gpd
from shapely import MultiPolygon

module_path = "/home/palmerjohn/code/dtc_ice_sheets/data_download_module/src/"
sys.path.insert(0, module_path)
module_path = "/home/palmerjohn/code/dtc_ice_sheets/data_converter/src"
sys.path.insert(0, module_path)

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    NormalisationType,
    TimeseriesRegion,
    TimeseriesType,
    UncertaintyType,
)

import yaml
from data_download_module.main import download_data
from data_download_module.schemas import DataDownloadRequest

import use_case_3.helper_functions as hf

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
            )

            timeseries.process_timeseries(
                data=summary_data,
                measurement_name=var,
                measurement_uncertainty_name=config_dict["timeseries_summary"][var]["measurement_uncertainties"],
                mask_geometries={"ice_shelf": None, "ocean": None, "climate": None},
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

