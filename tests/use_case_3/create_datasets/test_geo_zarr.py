"""Test the GeoZarr module."""

import os.path as path
import shutil

import numpy as np
import pandas as pd
import pytest

from use_case_3.create_datasets.create_dataset_helper_functions import *
from use_case_3.create_datasets.geo_zarr import GeoZarr


@pytest.fixture
def dataframe_fixture() -> pd.DataFrame:
    return pd.DataFrame(np.array([[8, 3, 0, 4], [10, 2, 5, 12], [9, 1, 11, 6], [7, 14, 15, 13], [19, 18, 17, 16]]),
                        columns=["y", "x", "z", "target"])


@pytest.fixture
def create_geozarr_memory_fixture() -> GeoZarr:

    return GeoZarr(storage_type=ZarrStorage.memory_store)

@pytest.fixture
def create_geozarr_local_store() -> GeoZarr:

    test_folder_output_path = path.join(path.dirname(__file__), "test_outputs", "geozarr_example.zarr")

    if path.isdir(test_folder_output_path):
        shutil.rmtree(test_folder_output_path)

    return GeoZarr(storage_type=ZarrStorage.local_store,
                   storage_directory=test_folder_output_path)

@pytest.fixture
def small_df_fixture() -> pd.DataFrame:

    data = {"x": [5, 1, 3, 2],
            "y": [5, 3, 2, 2],
            "value": [10, 5, 4, 7]}
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

    combined_tifs = return_common_elements_from_two_lists(list_1=elevation_tifs,
                                                          list_2=error_tifs,
                                                          ascending=True)

    data = load_tiffs2dataframe(file_names=combined_tifs,
                                folder_path_elevations=test_folder_input_path,
                                folder_path_errors=test_folder_input_path,
                                dem_diff=True)

    return data

def test__init___memory_store(create_geozarr_memory_fixture: GeoZarr):

    assert create_geozarr_memory_fixture.zarr_format == 2
    assert create_geozarr_memory_fixture.storage_type == ZarrStorage.memory_store
    assert create_geozarr_memory_fixture.storage_directory is None

def test__init__local_store(create_geozarr_local_store: GeoZarr):

    storage_directory = path.join(path.dirname(__file__), "test_outputs", "geozarr_example.zarr")
    assert create_geozarr_local_store.zarr_format == 2
    assert create_geozarr_local_store.storage_type == ZarrStorage.local_store
    assert create_geozarr_local_store.storage_directory == storage_directory

def test__reshape_data_simple(create_geozarr_local_store: GeoZarr):

    data = pd.DataFrame(np.array([[8, 3, 0, 4], [10, 2, 5, 12], [9, 1, 11, 6], [7, 14, 15, 13], [19, 18, 17, 16]]),
                        columns=["y", "x", "z", "target"])

    coord_columns = ["y", "x", "z"]
    coord_columns = sorted(coord_columns)

    # Order the columns alpabetically between the different column groups
    data = pd.concat([data["target"],
                        data[coord_columns]], axis=1)

    data = data.sort_values(coord_columns)

    assert data.iloc[0, 0] == 6
    assert data.iloc[2, 2] == 8
    assert data.iloc[4, 3] == 17

    data_reshaped, data_coordinates = create_geozarr_local_store._reshape_data(data=data,
                                                             target_column="target",
                                                             coord_columns=coord_columns)

    # Check the coordinate variables
    assert np.array_equal(data_coordinates[0], np.arange(1, 19))
    assert np.array_equal(data_coordinates[1], np.arange(7, 20))
    assert np.array_equal(data_coordinates[2], np.arange(0, 18))

    assert data_reshaped[0, 2, 11] == 6
    assert data_reshaped[1, 3, 5] == 12
    assert data_reshaped[2, 1, 0] == 4
    assert data_reshaped[13, 0, 15] == 13
    assert data_reshaped[17, 12, 17] == 16

def test__reshape_data_duplicate(create_geozarr_local_store: GeoZarr):

    data = pd.DataFrame(np.array([[8, 3, 0, 4],
                                  [10, 2, 5, 12],
                                  [9, 1, 11, 6],
                                  [7, 14, 15, 13],
                                  [19, 18, 17, 16],
                                  [9, 1, 17, 5]]),
                        columns=["y", "x", "z", "target"])

    coord_columns = ["y", "x", "z"]
    coord_columns = sorted(coord_columns)

    # Order the columns alpabetically between the different column groups
    data = pd.concat([data["target"],
                      data[coord_columns]], axis=1)

    data = data.sort_values(coord_columns)

    assert data.iloc[0, 0] == 6
    assert data.iloc[2, 2] == 10
    assert data.iloc[4, 3] == 15
    assert data.iloc[5, 3] == 17

    data_reshaped, data_coordinates = create_geozarr_local_store._reshape_data(data=data,
                                                             target_column="target",
                                                             coord_columns=coord_columns)

    # Check the coordinate variables
    assert np.array_equal(data_coordinates[0], np.arange(1, 19))
    assert np.array_equal(data_coordinates[1], np.arange(7, 20))
    assert np.array_equal(data_coordinates[2], np.arange(0, 18))

    assert data_reshaped[0, 2, 11] == 6
    assert data_reshaped[0, 2, 17] == 5
    assert data_reshaped[1, 3, 5] == 12
    assert data_reshaped[2, 1, 0] == 4
    assert data_reshaped[13, 0, 15] == 13
    assert data_reshaped[17, 12, 17] == 16

def test_create_dimensions_config(create_geozarr_local_store: GeoZarr):

    assert create_geozarr_local_store.dimensions_config.loc[0, "standard_name"] == "time"
    assert create_geozarr_local_store.dimensions_config.loc[3, "calendar"] == "None"

def test_calculate_chunk_dims(create_geozarr_local_store: GeoZarr):

    data_shape = (12, 12, 12)
    n_chunks = 3
    create_geozarr_local_store.n_chunks = n_chunks
    assert create_geozarr_local_store._calculate_chunk_dims(data_shape=data_shape) == (4, 4, 4)
    data_shape = (13, 13, 13)
    n_chunks = 3
    create_geozarr_local_store.n_chunks = n_chunks
    assert create_geozarr_local_store._calculate_chunk_dims(data_shape=data_shape) == (5, 5, 5)
    data_shape = (12, 12, 12)
    n_chunks = 1
    create_geozarr_local_store.n_chunks = n_chunks
    assert create_geozarr_local_store._calculate_chunk_dims(data_shape=data_shape) == (12, 12, 12)

def test_dataframe2geozarr(create_geozarr_local_store: GeoZarr,
                          example_df_fixture: pd.DataFrame):

    create_geozarr_local_store.dataframe2geozarr(data=example_df_fixture,
                                                 data_columns=["DEMdiff",
                                                               "error"],
                                                 coordinate_columns=["grid_longitude",
                                                                     "grid_latitude",
                                                                     "time"],
                                                 auxiliary_columns=[])

    assert sorted(create_geozarr_local_store.root.keys()) == ["DEMdiff",
                                                              "error",
                                                              "grid_latitude",
                                                              "grid_longitude",
                                                              "time"]

    assert create_geozarr_local_store.root["DEMdiff"][0, 0, 0] == 12.0
    assert create_geozarr_local_store.root["DEMdiff"][2, 0, 0] == 4
    assert create_geozarr_local_store.root["DEMdiff"][3, 2, 731] == 2
    assert np.isnan(create_geozarr_local_store.root["DEMdiff"][3, 2, 730])
    assert create_geozarr_local_store.root["DEMdiff"].attrs["_ARRAY_DIMENSIONS"] == ["grid_latitude",
                                                                                     "grid_longitude",
                                                                                     "time"]

    assert create_geozarr_local_store.root["grid_longitude"].attrs["_ARRAY_DIMENSIONS"] == ["grid_longitude"]
