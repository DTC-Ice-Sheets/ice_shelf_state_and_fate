"""Test the create dataset module."""

import os.path as path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import Polygon

from use_case_3.create_datasets.create_dataset_helper_functions import (
    calculate_coords_from_geotransform,
    convert_datetime_seconds,
    convert_decimal_dates,
    extract_date_string_from_filename,
    get_list_of_tiffs_from_folder,
    get_mask_from_shapefile,
    load_tiff2datacube,
    load_tiffs2dataframe,
    process_variable_datacube,
    process_variable_dataframe,
    return_common_elements_from_two_lists,
)


@pytest.fixture
def eolis_filenames_fixture():
    return "swath_uoe_v2_unc_20140415_dif.tif", "cryotempo_land_ice_ice_shelves_baseline_c_20100915_diff.tif"


@pytest.fixture
def gdal_geotransform_yneg_fixture():
    return "50.0 100.0 0.0 1450.0 0.0 -100.0", 4, 4


@pytest.fixture
def gdal_geotransform_ypos_fixture():
    return "50.0 100.0 0.0 1050.0 0.0 100.0", 4, 4


# def test_parse_dates_eolis_filename(eolis_filenames_fixture):
#     assert parse_dates_eolis_filename(eolis_filenames_fixture[0]) == "2014-04-15"
#     assert parse_dates_eolis_filename(eolis_filenames_fixture[1]) == "2010-09-15"


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
    eolis_filename_regex_pattern = r"(?:swath_uoe_v2_unc_|cryotempo_land_ice_ice_shelves_baseline_c_)(\d+)_.*\.tif"
    eolis_date_format = "%Y%m%d"

    assert (
        extract_date_string_from_filename(
            file_name="swath_uoe_v2_unc_20140415_dif.tif",
            file_name_regex_pattern=eolis_filename_regex_pattern,
            date_format=eolis_date_format,
        )
        == "2014-04-15"
    )

    assert (
        extract_date_string_from_filename(
            file_name="swath_uoe_v2_unc_20140415_dif.tif",
            file_name_regex_pattern=eolis_filename_regex_pattern,
            date_format=eolis_date_format,
        )
        == "2014-04-15"
    )

    assert (
        extract_date_string_from_filename(
            file_name="2015-01-01.tif", file_name_regex_pattern=r"(\d+-\d+-\d+)\.tif", date_format="%Y-%m-%d"
        )
        == "2015-01-01"
    )
    # assert extract_date_string_from_filename("2015-01-01.tif") == "2015-01-01"


def test_load_tiffs2datacube():
    test_folder_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    # It doesn't matter that these are the same for testing
    elevation_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)
    error_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)

    combined_tifs = return_common_elements_from_two_lists(list_1=elevation_tifs, list_2=error_tifs, ascending=True)

    reference_time = pd.Timestamp(year=1970, month=1, day=1)

    file_name_regex_pattern = r"(\d+-\d+-\d+)\.tif"
    date_format = "%Y-%m-%d"

    data, data_errors, data_coords, mask_raster = load_tiff2datacube(
        file_names=combined_tifs,
        folder_path_observations=test_folder_path,
        reference_time=reference_time,
        error_file_names=combined_tifs,
        folder_path_errors=test_folder_path,
        file_path_mask=None,
        create_raster_mask=False,
        geometry_name=None,
        process_datetimes=True,
        file_name_regex_pattern=file_name_regex_pattern,
        date_format=date_format,
    )

    assert data.shape == (3, 4, 4)
    assert data[0, 0, 0] == 0
    assert data[2, 0, 0] == 0
    assert data[0, 3, 3] == 15
    assert data[1, 3, 3] == 15
    assert data_errors.shape == (3, 4, 4)
    assert data_errors[0, 0, 0] == 0
    assert data_errors[2, 0, 0] == 0
    assert data_errors[0, 3, 3] == 15
    assert data_errors[1, 3, 3] == 15
    assert mask_raster is None
    assert np.array_equal(data_coords["projection_y_coordinate"], np.array([1450, 1350, 1250, 1150]))
    assert np.array_equal(data_coords["projection_x_coordinate"], np.array([50, 150, 250, 350]))
    assert np.array_equal(
        data_coords["time"],
        np.array(
            [
                (pd.Timestamp(year=2015, month=1, day=1) - reference_time).total_seconds(),
                (pd.Timestamp(year=2016, month=1, day=1) - reference_time).total_seconds(),
                (pd.Timestamp(year=2017, month=1, day=1) - reference_time).total_seconds(),
            ]
        ),
    )


def test_process_variable_datacube():
    test_folder_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    # It doesn't matter that these are the same for testing
    elevation_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)
    combined_tifs = return_common_elements_from_two_lists(list_1=elevation_tifs, list_2=[], ascending=True)

    reference_time = pd.Timestamp(year=1970, month=1, day=1)
    file_name_regex_pattern = r"(\d+-\d+-\d+)\.tif"
    date_format = "%Y-%m-%d"

    data, data_coords, mask_raster = process_variable_datacube(
        file_names=combined_tifs,
        folder_path_variable=test_folder_path,
        reference_time=reference_time,
        file_path_mask=None,
        create_raster_mask=False,
        geometry_name=None,
        process_datetimes=True,
        file_name_regex_pattern=file_name_regex_pattern,
        date_format=date_format,
    )

    assert data.shape == (3, 4, 4)
    assert data[0, 0, 0] == 0
    assert data[2, 0, 0] == 0
    assert data[0, 3, 3] == 15
    assert data[1, 3, 3] == 15
    assert mask_raster is None
    assert np.array_equal(data_coords["projection_y_coordinate"], np.array([1450, 1350, 1250, 1150]))
    assert np.array_equal(data_coords["projection_x_coordinate"], np.array([50, 150, 250, 350]))
    assert np.allclose(
        data_coords["time"],
        np.array(
            [
                (pd.Timestamp(year=2015, month=1, day=1) - reference_time).total_seconds(),
                (pd.Timestamp(year=2016, month=1, day=1) - reference_time).total_seconds(),
                (pd.Timestamp(year=2017, month=1, day=1) - reference_time).total_seconds(),
            ]
        ),
    )


def test_load_tiffs2dataframe():
    test_folder_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    # It doesn't matter that these are the same for testing
    elevation_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)
    error_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)

    combined_tifs = return_common_elements_from_two_lists(list_1=elevation_tifs, list_2=error_tifs, ascending=True)

    coord_names = ["projection_x_coordinate", "projection_y_coordinate", "time"]
    reference_time = pd.Timestamp(year=1970, month=1, day=1)
    observation_names = {
        "value": "height_difference_from_reference_elevation",
        "x": "projection_x_coordinate",
        "y": "projection_y_coordinate",
    }

    file_name_regex_pattern = r"(\d+-\d+-\d+)\.tif"
    date_format = "%Y-%m-%d"

    data, mask_raster = load_tiffs2dataframe(
        file_names=combined_tifs,
        folder_path_observations=test_folder_path,
        observation_names=observation_names,
        coord_names=coord_names,
        reference_time=reference_time,
        error_file_names=combined_tifs,
        folder_path_errors=test_folder_path,
        file_path_mask=None,
        create_raster_mask=False,
        geometry_name=None,
        process_datetimes=True,
        file_name_regex_pattern=file_name_regex_pattern,
        date_format=date_format,
    )

    assert data.shape == (48, 5)
    assert list(data.columns) == [
        "projection_x_coordinate",
        "projection_y_coordinate",
        "time",
        "height_difference_from_reference_elevation",
        "height_difference_from_reference_elevation_uncertainty",
    ]
    assert mask_raster is None


def test_process_variable_dataframe():
    test_folder_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    # It doesn't matter that these are the same for testing
    elevation_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)
    error_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_path)

    combined_tifs = return_common_elements_from_two_lists(list_1=elevation_tifs, list_2=error_tifs, ascending=True)

    coord_names = ["projection_x_coordinate", "projection_y_coordinate", "time"]
    reference_time = pd.Timestamp(year=1970, month=1, day=1)
    observation_names = {
        "value": "height_difference_from_reference_elevation",
        "x": "projection_x_coordinate",
        "y": "projection_y_coordinate",
    }

    file_name_regex_pattern = r"(\d+-\d+-\d+)\.tif"
    date_format = "%Y-%m-%d"

    data, mask_raster = process_variable_dataframe(
        file_names=combined_tifs,
        folder_path_variable=test_folder_path,
        variable_names=observation_names,
        coord_names=coord_names,
        reference_time=reference_time,
        file_path_mask=None,
        create_raster_mask=False,
        geometry_name=None,
        process_datetimes=True,
        file_name_regex_pattern=file_name_regex_pattern,
        date_format=date_format,
    )

    assert data.shape == (48, 4)
    assert list(data.columns) == [
        "projection_x_coordinate",
        "projection_y_coordinate",
        "time",
        "height_difference_from_reference_elevation",
    ]
    assert mask_raster is None


def test_get_mask_from_shapefile():
    # Need to create a mask to load from a .gpkg
    test_mask_input_path = path.join(path.dirname(__file__), "test_inputs", "toy_mask_4x4.gpkg")

    coords = ((100.0, 1200.0), (300.0, 1200.0), (300.0, 1300.0), (100.0, 1300.0))
    polygon = Polygon(coords)

    d = {"Name": ["test"], "geometry": [polygon]}

    gdf = gpd.GeoDataFrame(d)
    gdf.to_file(test_mask_input_path)

    test_file_path = path.join(path.dirname(__file__), "test_inputs", "rasters", "2015-01-01.tif")

    data_mask = get_mask_from_shapefile(
        file_path_raster=test_file_path, file_path_mask=test_mask_input_path, geometry_name="test"
    )

    assert data_mask[0, 0]
    assert data_mask[0, 1]
    assert data_mask[0, 2]
    assert data_mask[0, 3]
    assert data_mask[1, 3]
    assert data_mask[2, 3]
    assert data_mask[3, 0]
    assert data_mask[3, 1]
    assert data_mask[3, 2]
    assert data_mask[3, 3]

    assert not data_mask[1, 0]
    assert not data_mask[1, 1]
    assert not data_mask[1, 2]
    assert not data_mask[2, 0]
    assert not data_mask[2, 1]
    assert not data_mask[2, 2]


def test_convert_decimal_date():
    decimal_dates = pd.Series([2012.0, 2012.5, 2013.5, 2014.5])
    timestamps = convert_decimal_dates(decimal_dates=decimal_dates)

    expected_timestamps = pd.Series(
        [
            pd.Timestamp(year=2012, month=1, day=1, hour=0),
            pd.Timestamp(year=2012, month=7, day=2, hour=0),
            pd.Timestamp(year=2013, month=7, day=2, hour=12),
            pd.Timestamp(year=2014, month=7, day=2, hour=12),
        ],
        name="datetime",
    )

    assert timestamps.equals(expected_timestamps)


def test_convert_datetime_seconds():
    reference = pd.Timestamp(year=1970, month=1, day=1, hour=0, minute=0, second=0)
    time_series = pd.Series(
        [
            pd.Timestamp(year=1970, month=1, day=1, hour=0, minute=0, second=0),
            pd.Timestamp(year=1970, month=1, day=1, hour=0, minute=0, second=1),
            pd.Timestamp(year=1970, month=1, day=1, hour=0, minute=0, second=2),
        ]
    )

    pd.testing.assert_series_equal(
        convert_datetime_seconds(time_series, reference),
        pd.Series(
            [
                0,
                1,
                2,
            ]
        ).astype(float),
    )


def test_calculate_coords_from_geotransform(gdal_geotransform_yneg_fixture, gdal_geotransform_ypos_fixture):
    y_coord, x_coord = calculate_coords_from_geotransform(
        geotransform=gdal_geotransform_yneg_fixture[0],
        x_dim=gdal_geotransform_yneg_fixture[1],
        y_dim=gdal_geotransform_yneg_fixture[2],
    )

    assert np.array_equal(y_coord, np.array([1450, 1350, 1250, 1150]))
    assert np.array_equal(x_coord, np.array([50, 150, 250, 350]))

    y_coord, x_coord = calculate_coords_from_geotransform(
        geotransform=gdal_geotransform_ypos_fixture[0],
        x_dim=gdal_geotransform_ypos_fixture[1],
        y_dim=gdal_geotransform_ypos_fixture[2],
    )

    assert np.array_equal(y_coord, np.array([1050, 1150, 1250, 1350]))
    assert np.array_equal(x_coord, np.array([50, 150, 250, 350]))
