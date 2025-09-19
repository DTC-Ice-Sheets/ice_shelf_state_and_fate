import sys

import geopandas as gpd
from shapely import MultiPolygon

module_path = "/home/palmerjohn/code/dtc_ice_sheets/data_download_module/src/"
sys.path.insert(0, module_path)
module_path = "/home/palmerjohn/code/dtc_ice_sheets/data_converter/src"
sys.path.insert(0, module_path)

from pathlib import Path

import xarray as xr

# from use_case_3.timeseries_module.timeseries import Timeseries
# from use_case_3.timeseries_module.timeseries_enums import (
#     AggregationStrategy,
#     AggregationPeriod,
#     UncertaintyType,
#     NormalisationType,
#     # PlotSpreadType,
#     TimeseriesType,
# )
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
    ice_shelf_polygons: gpd.GeoDataFrame, ice_shelf: str = "Thwaites"
) -> tuple[xr.Dataset, MultiPolygon, MultiPolygon]:
    ice_shelf_polygon_epsg4326, ice_shelf_polygon = hf.iceshelf_polygon(
        ice_shelf_polygons=ice_shelf_polygons, ice_shelf_name=ice_shelf
    )

    local_raw_request = DataDownloadRequest(dataset_id=f"{ice_shelf}_SUMMARY".upper(), raw_download=False)

    summary_data_path = download_data(data_download_request=local_raw_request, output_parent_directory=OUTPUT_DIRECTORY)

    summary_data = xr.open_zarr(store=summary_data_path, zarr_format=2)

    return summary_data, ice_shelf_polygon_epsg4326, ice_shelf_polygon
