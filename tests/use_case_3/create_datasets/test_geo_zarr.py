"""Test the GeoZarr module."""

import os.path as path
import shutil

import numpy as np
import pandas as pd
import pytest

from use_case_3.create_datasets.create_dataset_helper_functions import *
from use_case_3.create_datasets.geo_zarr import GeoZarr


@pytest.fixture
def create_geozarr_memory_fixture() -> GeoZarr:

    return GeoZarr(storage_type="memory_store")

@pytest.fixture
def create_geozarr_local_store() -> GeoZarr:

    test_folder_output_path = path.join(path.dirname(__file__), "test_outputs", "geozarr_example.zarr")

    if path.isdir(test_folder_output_path):
        shutil.rmtree(test_folder_output_path)

    return GeoZarr(storage_type="local_store",
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
    assert create_geozarr_memory_fixture.storage_type == "memory_store"
    assert create_geozarr_memory_fixture.storage_directory is None

def test__init__local_store(create_geozarr_local_store: GeoZarr):

    storage_directory = path.join(path.dirname(__file__), "test_outputs", "geozarr_example.zarr")
    assert create_geozarr_local_store.zarr_format == 2
    assert create_geozarr_local_store.storage_type == "local_store"
    assert create_geozarr_local_store.storage_directory == storage_directory


def test__reshape_data(example_df_fixture: pd.DataFrame,
                       create_geozarr_local_store: GeoZarr):

    n_random_checks = 2

    coord_columns = ["grid_latitude", "grid_longitude", "time"]

    example_df_fixture = example_df_fixture.sort_values(coord_columns)

    data_reshaped = create_geozarr_local_store._reshape_data(data=example_df_fixture,
                                                             target_column="DEMdiff",
                                                             coord_columns=coord_columns)

    latitudes = create_geozarr_local_store._reduce_coordinates(data=example_df_fixture,
                                                               target_column=coord_columns[0])

    longitudes = create_geozarr_local_store._reduce_coordinates(data=example_df_fixture,
                                                               target_column=coord_columns[1])

    times = create_geozarr_local_store._reduce_coordinates(data=example_df_fixture,
                                                               target_column=coord_columns[2])

    for _ in range(n_random_checks):
        idx  = np.random.choice(example_df_fixture.index)
        lat = example_df_fixture.loc[idx, coord_columns[0]]
        long = example_df_fixture.loc[idx, coord_columns[1]]
        time = example_df_fixture.loc[idx, coord_columns[2]]
        value_ground_truth = example_df_fixture.loc[idx, "DEMdiff"]

        value_reshape = np.squeeze(data_reshaped[np.where(latitudes == lat),
                                       np.where(longitudes == long),
                                       np.where(times == time)])

        assert value_ground_truth == value_reshape

def test_reduce_coordinates(small_df_fixture: pd.DataFrame):

    assert np.array_equal(GeoZarr._reduce_coordinates(data=small_df_fixture, target_column="x"), np.array([1, 2, 3, 5]))
    assert np.array_equal(GeoZarr._reduce_coordinates(data=small_df_fixture, target_column="y"), np.array([2, 3, 5]))

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

def test_dataframe2geozar(create_geozarr_local_store: GeoZarr,
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
    assert create_geozarr_local_store.root["DEMdiff"][2, 3, 1] == 7.0
    assert create_geozarr_local_store.root["DEMdiff"].attrs["_ARRAY_DIMENSIONS"] == ["grid_latitude",
                                                                                     "grid_longitude",
                                                                                     "time"]

    assert create_geozarr_local_store.root["grid_longitude"].attrs["_ARRAY_DIMENSIONS"] == ["grid_longitude"]
