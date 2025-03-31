"""Test the create dataset module."""

import os.path as path

import geopandas as gpd
from shapely import Polygon

from use_case_3.create_datasets.create_dataset_helper_functions import (
    extract_date_string_from_filename,
    get_list_of_tiffs_from_folder,
    get_mask_from_shapefile,
    load_tiffs2dataframe,
    load_tiffs2dataframe_basalmelt,
    return_common_elements_from_two_lists,
)
from use_case_3.raster_tools.raster_np import RasterNp


def test_get_list_of_tifs_from_folder():
    test_folder_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    expected_result_1 = {"2015-01-01.tif", "2016-01-01.tif", "2017-01-01.tif"}
    expected_result_2 = {"2015-01-01.tif", "2016-01-01.tif", "2017-01-01.tif", "2017-01-01.jpg"}
    assert set(get_list_of_tiffs_from_folder(folder_path=test_folder_path)) == expected_result_1
    assert set(get_list_of_tiffs_from_folder(folder_path=test_folder_path)) != expected_result_2


def test_return_common_elements_from_two_lists():
    list_1 = ["item_3", "item_2", "item_1", "item_1", "item_4"]
    list_2 = ["item_0", "item_2", "item_1", "item_3", "item_4"]

    expected_result_1 = ["item_1", "item_2", "item_3", "item_4"]
    assert return_common_elements_from_two_lists(list_1, list_2, ascending=True) == expected_result_1

    expected_result_2 = ["item_4", "item_3", "item_2", "item_1"]
    assert return_common_elements_from_two_lists(list_1, list_2, ascending=False) == expected_result_2


def test_extract_date_string_from_filename():
    assert extract_date_string_from_filename("2015-01-01.tif") == "2015-01-01"


def test_load_tiffs2dataframe():
    test_folder_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    # It doesn't matter that these are the same for testing
    elevation_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)
    error_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)

    combined_tifs = return_common_elements_from_two_lists(list_1=elevation_tifs, list_2=error_tifs, ascending=True)

    data = load_tiffs2dataframe(
        file_names=combined_tifs,
        folder_path_elevations=test_folder_path,
        folder_path_errors=test_folder_path,
        dem_diff=True,
    )

    assert data.shape == (48, 5)
    assert list(data.columns) == [
        "projection_x_coordinate",
        "projection_y_coordinate",
        "time",
        "height_difference_from_dem",
        "height_difference_from_dem_uncertainty",
    ]

    data = load_tiffs2dataframe(
        file_names=combined_tifs,
        folder_path_elevations=test_folder_path,
        folder_path_errors=test_folder_path,
        dem_diff=False,
    )

    assert data.shape == (48, 5)
    assert list(data.columns) == [
        "projection_x_coordinate",
        "projection_y_coordinate",
        "time",
        "height_above_mean_sea_level",
        "height_above_mean_sea_level_uncertainty",
    ]


def test_load_tiffs2dataframe_basalmelt():
    # It doesn't matter that these are the same for testing
    test_file_path = path.join(path.dirname(__file__), "test_inputs", "rasters", "2015-01-01.tif")

    data = load_tiffs2dataframe_basalmelt(file_path_obvs=test_file_path, file_path_errors=test_file_path)

    assert data.shape == (16, 4)
    assert list(data.columns) == [
        "projection_x_coordinate",
        "projection_y_coordinate",
        "basal_melt_rate",
        "basal_melt_rate_uncertainty",
    ]


def test_get_mask_from_shapefile():
    # Need to create a mask to load from a .gpkg
    test_mask_input_path = path.join(path.dirname(__file__), "test_inputs", "masks", "toy_mask_4x4.gpkg")

    coords = ((100., 1200.), ( 300., 1200.), (300., 1300.), (100., 1300.))
    polygon = Polygon(coords)

    d =  {"Name": ["test"], "geometry": [polygon]}

    gdf = gpd.GeoDataFrame(d)
    gdf.to_file(test_mask_input_path)

    test_file_path = path.join(path.dirname(__file__), "test_inputs", "rasters", "2015-01-01.tif")

    data = RasterNp.load_from_tiff(file_path=test_file_path)

    data_mask = get_mask_from_shapefile(
        file_path_raster=test_file_path, file_path_mask=test_mask_input_path, glacier_name="test"
    )

    assert data_mask[0, 0]
    assert data_mask[3, 3]
    assert data_mask[3, 1]

    assert not data_mask[1, 1]
    assert not data_mask[2, 2]
