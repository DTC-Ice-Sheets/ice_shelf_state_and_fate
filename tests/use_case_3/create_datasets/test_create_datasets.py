"""Test the create dataset module."""

import os.path as path

import pandas as pd
import pytest

from use_case_3.create_datasets.create_datasets import CreateDataSets


@pytest.fixture
def init_create_datasets() -> CreateDataSets:
    """
    Initialse the CreateDataSets object.

    Returns
    -------
    CreateDataSets
        A CreateDataSet object.
    """
    return CreateDataSets()

def test_get_list_of_tifs_from_folder():

    test_folder_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    expected_result_1 = {"2015-01-01.tif", "2016-01-01.tif", "2017-01-01.tif"}
    expected_result_2 = {"2015-01-01.tif", "2016-01-01.tif", "2017-01-01.tif", "2017-01-01.jpg"}
    assert set(CreateDataSets.get_list_of_tiffs_from_folder(folder_path=test_folder_path)) == expected_result_1
    assert set(CreateDataSets.get_list_of_tiffs_from_folder(folder_path=test_folder_path)) != expected_result_2

def test_return_common_elements_from_two_lists():

    list_1 = ["item_3", "item_2", "item_1", "item_1", "item_4"]
    list_2 = ["item_0", "item_2", "item_1", "item_3", "item_4"]

    expected_result_1 = ["item_1", "item_2", "item_3", "item_4"]
    assert CreateDataSets.return_common_elements_from_two_lists(list_1, list_2, ascending=True) == expected_result_1

    expected_result_2 = ["item_4", "item_3", "item_2", "item_1"]
    assert CreateDataSets.return_common_elements_from_two_lists(list_1, list_2, ascending=False) == expected_result_2

def test_convert_filename2date():

    assert CreateDataSets.convert_filename2date("2015-01-01.tif") == "2015-01-01"

def test_load_tiffs2dataframe(init_create_datasets: CreateDataSets):

    test_folder_path = path.join(path.dirname(__file__), "test_inputs", "rasters")

    # It doesn't matter that these are the same for testing
    elevation_tifs = init_create_datasets.get_list_of_tiffs_from_folder(folder_path=test_folder_path)
    error_tifs = init_create_datasets.get_list_of_tiffs_from_folder(folder_path=test_folder_path)

    combined_tifs = init_create_datasets.return_common_elements_from_two_lists(list_1=elevation_tifs,
                                                                               list_2=error_tifs,
                                                                               ascending=True)

    data = init_create_datasets.load_tiffs2dataframe(file_names=combined_tifs,
                                                     folder_path_elevations=test_folder_path,
                                                     folder_path_errors=test_folder_path,
                                                     dem_diff=True)

    assert data.shape == (48, 5)
    assert list(data.columns) == ["x", "y", "date", "DEMdiff (m)", "error (m)"]

    data = init_create_datasets.load_tiffs2dataframe(file_names=combined_tifs,
                                                     folder_path_elevations=test_folder_path,
                                                     folder_path_errors=test_folder_path,
                                                     dem_diff=False)

    assert data.shape == (48, 5)
    assert list(data.columns) == ["x", "y", "date", "elevation (m)", "error (m)"]
