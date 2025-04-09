"""Test the GeoZarr module."""

import os.path as path
import shutil

import numpy as np
import pandas as pd
import pytest

from use_case_3.create_datasets.create_dataset_helper_functions import (
    get_list_of_tiffs_from_folder,
    load_tiffs2dataframe,
    return_common_elements_from_two_lists,
)
from use_case_3.create_datasets.geo_zarr import GeoZarr, ZarrStorage


@pytest.fixture
def dataframe_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        np.array([[8, 3, 0, 4], [10, 2, 5, 12], [9, 1, 11, 6], [7, 14, 15, 13], [19, 18, 17, 16]]),
        columns=["projection_y_coordinate", "projection_x_coordinate", "time", "target"],
    )


@pytest.fixture
def geozarr_memory_store() -> GeoZarr:
    return GeoZarr(storage_type=ZarrStorage.memory_store)


@pytest.fixture
def geozarr_local_store() -> GeoZarr:
    test_folder_output_path = path.join(path.dirname(__file__), "test_outputs", "geozarr_example.zarr")

    if path.isdir(test_folder_output_path):
        shutil.rmtree(test_folder_output_path)

    return GeoZarr(storage_type=ZarrStorage.local_store, storage_directory=test_folder_output_path)


@pytest.fixture
def small_df_fixture() -> pd.DataFrame:
    data = {"projection_x_coordinate": [5, 1, 3, 2], "projection_y_coordinate": [5, 3, 2, 2], "value": [10, 5, 4, 7]}
    return pd.DataFrame(data)


@pytest.fixture
def example_df_fixture() -> pd.DataFrame:
    test_folder_input_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    test_folder_output_path = path.join(path.dirname(__file__), "test_outputs", "geozarr_example.zarr")

    if path.isdir(test_folder_output_path):
        shutil.rmtree(test_folder_output_path)

    # It doesn't matter that these are the same for testing
    elevation_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_input_path)
    error_tifs = get_list_of_tiffs_from_folder(folder_path=test_folder_input_path)

    combined_tifs = return_common_elements_from_two_lists(list_1=elevation_tifs, list_2=error_tifs, ascending=True)

    data = load_tiffs2dataframe(
        file_names=combined_tifs,
        folder_path_elevations=test_folder_input_path,
        folder_path_errors=test_folder_input_path,
        dem_diff=True,
    )

    return data


def test__init___memory_store(geozarr_memory_store: GeoZarr):
    assert geozarr_memory_store.zarr_format == 2
    assert geozarr_memory_store.storage_type == ZarrStorage.memory_store
    assert geozarr_memory_store.storage_directory is None


def test__init__local_store(geozarr_local_store: GeoZarr):
    storage_directory = path.join(path.dirname(__file__), "test_outputs", "geozarr_example.zarr")
    assert geozarr_local_store.zarr_format == 2
    assert geozarr_local_store.storage_type == ZarrStorage.local_store
    assert geozarr_local_store.storage_directory == storage_directory


def test__reshape_data_simple(geozarr_local_store: GeoZarr):
    data = pd.DataFrame(
        np.array([[8, 3, 0, 4], [10, 2, 5, 12], [9, 1, 11, 6], [7, 14, 15, 13], [19, 18, 17, 16]]),
        columns=["projection_y_coordinate", "projection_x_coordinate", "time", "target"],
    )

    coord_columns = ["projection_y_coordinate", "projection_x_coordinate", "time"]

    assert data.iloc[0, 0] == 8
    assert data.iloc[2, 2] == 11
    assert data.iloc[4, 3] == 16

    data_reshaped, data_coordinates = geozarr_local_store._reshape_data(
        data=data, target_column="target", coord_columns=coord_columns, y_affine_negative=False
    )

    # Check the coordinate variables
    # assert np.array_equal(data_coordinates[1], np.arange(19, 6, -1))
    assert np.array_equal(data_coordinates[0], np.arange(7, 20))
    assert np.array_equal(data_coordinates[1], np.arange(1, 19))
    assert np.array_equal(data_coordinates[2], np.array([0, 5, 11, 15, 17]))

    assert data_reshaped[1, 2, 0] == 4
    assert data_reshaped[3, 1, 1] == 12
    assert data_reshaped[2, 0, 2] == 6
    assert data_reshaped[0, 13, 3] == 13
    assert data_reshaped[12, 17, 4] == 16


def test__reshape_data_simple_affine_negative(geozarr_local_store: GeoZarr):
    data = pd.DataFrame(
        np.array([[8, 3, 0, 4], [10, 2, 5, 12], [9, 1, 11, 6], [7, 14, 15, 13], [19, 18, 17, 16]]),
        columns=["projection_y_coordinate", "projection_x_coordinate", "time", "target"],
    )

    coord_columns = ["projection_y_coordinate", "projection_x_coordinate", "time"]

    assert data.iloc[0, 0] == 8
    assert data.iloc[2, 2] == 11
    assert data.iloc[4, 3] == 16

    data_reshaped, data_coordinates = geozarr_local_store._reshape_data(
        data=data, target_column="target", coord_columns=coord_columns, y_affine_negative=True
    )

    # Check the coordinate variables
    assert np.array_equal(data_coordinates[0], np.arange(19, 6, -1))
    assert np.array_equal(data_coordinates[1], np.arange(1, 19))
    assert np.array_equal(data_coordinates[2], np.array([0, 5, 11, 15, 17]))

    assert data_reshaped[11, 2, 0] == 4
    assert data_reshaped[9, 1, 1] == 12
    assert data_reshaped[10, 0, 2] == 6
    assert data_reshaped[12, 13, 3] == 13
    assert data_reshaped[0, 17, 4] == 16


def test__reshape_data_duplicate(geozarr_local_store: GeoZarr):
    data = pd.DataFrame(
        np.array([[8, 3, 0, 4], [10, 2, 5, 12], [9, 1, 11, 6], [7, 14, 15, 13], [19, 18, 17, 16], [9, 1, 17, 5]]),
        columns=["projection_y_coordinate", "projection_x_coordinate", "time", "target"],
    )

    coord_columns = ["projection_y_coordinate", "projection_x_coordinate", "time"]
    # coord_columns = sorted(coord_columns)

    # # Order the columns alpabetically between the different column groups
    # data = pd.concat([data["target"], data[coord_columns]], axis=1)

    # data = data.sort_values(coord_columns)

    assert data.iloc[0, 0] == 8
    assert data.iloc[2, 2] == 11
    assert data.iloc[4, 3] == 16
    assert data.iloc[5, 3] == 5

    data_reshaped, data_coordinates = geozarr_local_store._reshape_data(
        data=data, target_column="target", coord_columns=coord_columns, y_affine_negative=False
    )

    # Check the coordinate variables
    assert np.array_equal(data_coordinates[0], np.arange(7, 20))
    assert np.array_equal(data_coordinates[1], np.arange(1, 19))
    assert np.array_equal(data_coordinates[2], np.array([0, 5, 11, 15, 17]))

    assert data_reshaped[1, 2, 0] == 4
    assert data_reshaped[3, 1, 1] == 12
    assert data_reshaped[2, 0, 2] == 6
    assert data_reshaped[0, 13, 3] == 13
    assert data_reshaped[12, 17, 4] == 16
    assert data_reshaped[2, 0, 4] == 5


def test_create_dimensions_config(geozarr_local_store: GeoZarr):
    assert geozarr_local_store.dimensions_config.loc[0, "standard_name"] == "time"
    assert geozarr_local_store.dimensions_config.loc[3, "calendar"] == "None"


@pytest.mark.parametrize(
    "data_shape,n_chunks, chunk_shape",
    [((12, 12, 12), 3, (4, 4, 4)), ((13, 13, 13), 3, (5, 5, 5)), ((12, 12, 12), 1, (12, 12, 12))],
)
def test_calculate_chunk_dims(geozarr_local_store: GeoZarr, data_shape, n_chunks, chunk_shape):
    geozarr_local_store.n_chunks = n_chunks
    assert geozarr_local_store._calculate_chunk_dims(data_shape=data_shape) == chunk_shape


def test_dataframe2geozarr(geozarr_local_store: GeoZarr, example_df_fixture: pd.DataFrame):
    geozarr_local_store.dataframe2geozarr(
        data=example_df_fixture,
        data_columns=["height_difference_from_dem", "height_difference_from_dem_uncertainty"],
        coordinate_columns=["projection_y_coordinate", "projection_x_coordinate", "time"],
        auxiliary_columns=[],
        y_affine_negative=False,
    )

    # Check that the variables have all been stored in the root zarr group, need to sort these as
    # Zarr stores these in a dictionary which will be returned in a random order.
    assert sorted(geozarr_local_store.root.keys()) == sorted(
        [
            "time",
            "height_difference_from_dem",
            "height_difference_from_dem_uncertainty",
            "projection_y_coordinate",
            "projection_x_coordinate",
        ]
    )

    assert geozarr_local_store.root["height_difference_from_dem"][0, 0, 0] == 12
    assert geozarr_local_store.root["height_difference_from_dem"][2, 0, 0] == 4
    assert geozarr_local_store.root["height_difference_from_dem"][3, 2, 2] == 2
    assert geozarr_local_store.root["height_difference_from_dem"].attrs["_ARRAY_DIMENSIONS"] == [
        "projection_y_coordinate",
        "projection_x_coordinate",
        "time",
    ]

    assert geozarr_local_store.root["projection_y_coordinate"].attrs["_ARRAY_DIMENSIONS"] == ["projection_y_coordinate"]


def test_dataframe2geozarr_affine_negative(geozarr_local_store: GeoZarr, example_df_fixture: pd.DataFrame):
    geozarr_local_store.dataframe2geozarr(
        data=example_df_fixture,
        data_columns=["height_difference_from_dem", "height_difference_from_dem_uncertainty"],
        coordinate_columns=["projection_y_coordinate", "projection_x_coordinate", "time"],
        auxiliary_columns=[],
        y_affine_negative=True,
    )

    # Check that the variables have all been stored in the root zarr group, need to sort these as
    # Zarr stores these in a dictionary which will be returned in a random order.
    assert sorted(geozarr_local_store.root.keys()) == sorted(
        [
            "time",
            "height_difference_from_dem",
            "height_difference_from_dem_uncertainty",
            "projection_y_coordinate",
            "projection_x_coordinate",
        ]
    )

    assert geozarr_local_store.root["height_difference_from_dem"][3, 0, 0] == 12
    assert geozarr_local_store.root["height_difference_from_dem"][1, 0, 0] == 4
    assert geozarr_local_store.root["height_difference_from_dem"][0, 2, 2] == 2
    assert geozarr_local_store.root["height_difference_from_dem"].attrs["_ARRAY_DIMENSIONS"] == [
        "projection_y_coordinate",
        "projection_x_coordinate",
        "time",
    ]

    assert geozarr_local_store.root["projection_y_coordinate"].attrs["_ARRAY_DIMENSIONS"] == ["projection_y_coordinate"]


def test___convert_x_coord_name():
    assert GeoZarr._convert_x_coord_name("X") == "projection_x_coordinate"
    assert GeoZarr._convert_x_coord_name("x") == "projection_x_coordinate"
    assert GeoZarr._convert_x_coord_name("xy") is None


def test___convert_y_coord_name():
    assert GeoZarr._convert_y_coord_name("Y") == "projection_y_coordinate"
    assert GeoZarr._convert_y_coord_name("y") == "projection_y_coordinate"
    assert GeoZarr._convert_y_coord_name("xy") is None


def test__convert_variable_name(geozarr_local_store):
    assert geozarr_local_store._convert_variable_name("X") == "projection_x_coordinate"
    assert geozarr_local_store._convert_variable_name("Y") == "projection_y_coordinate"
    assert geozarr_local_store._convert_variable_name("XY") == "XY"


def test_convert_netcdf_to_geo_zarr(geozarr_local_store):
    net_cdf_path = path.join(path.dirname(__file__), "test_inputs", "netcdf_example.nc")
    geozarr_local_store.convert_netcdf_to_geo_zarr(
        filepath=net_cdf_path, target_columns=["test"], coordinate_columns=["y", "x"]
    )

    assert np.array_equal(geozarr_local_store.root["projection_x_coordinate"], np.array([0, 1, 2]))
    assert np.array_equal(geozarr_local_store.root["projection_y_coordinate"], np.array([0, 1, 2]))
    assert np.array_equal(geozarr_local_store.root["test"], np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert geozarr_local_store.root.attrs["_ARRAY_DIMENSIONS"] == ["projection_y_coordinate", "projection_x_coordinate"]
    assert geozarr_local_store.root["projection_x_coordinate"].attrs["_ARRAY_DIMENSIONS"] == ["projection_x_coordinate"]
    assert geozarr_local_store.root["projection_y_coordinate"].attrs["_ARRAY_DIMENSIONS"] == ["projection_y_coordinate"]
    assert geozarr_local_store.root["test"].attrs["_ARRAY_DIMENSIONS"] == [
        "projection_y_coordinate",
        "projection_x_coordinate",
    ]


def test_create_coordinate_reference_system(geozarr_local_store):
    crs = geozarr_local_store.create_coordinate_reference_system(
        grid_mapping_name="test",
        straight_vertical_longitude_from_pole=None,
        latitude_of_projection_origin=None,
        false_easting=None,
        false_northing=None,
        semi_major_axis=None,
        inverse_flattening=None,
        geotransform=None,
        srid=None,
        proj4text=None,
        standard_parallel=None,
        crs_wkt=None,
        add_attributes=False,
    )
    assert crs == {"grid_mapping_name": "test"}
    assert "_CRS" not in geozarr_local_store.root.attrs.keys()

    crs = geozarr_local_store.create_coordinate_reference_system(
        grid_mapping_name="test",
        straight_vertical_longitude_from_pole=None,
        latitude_of_projection_origin=None,
        false_easting=None,
        false_northing=None,
        semi_major_axis=None,
        inverse_flattening=None,
        geotransform=None,
        srid=None,
        proj4text=None,
        standard_parallel=None,
        crs_wkt=None,
        add_attributes=True,
    )
    assert crs == {"grid_mapping_name": "test"}
    assert geozarr_local_store.root.attrs["_CRS"] == {"grid_mapping_name": "test"}
