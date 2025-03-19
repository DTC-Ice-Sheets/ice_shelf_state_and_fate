"""Test the create dataset module."""

import os.path as path

import pytest

from use_case_3.create_datasets.create_datasets import CreateDataSets


def init_create_datasets() -> CreateDataSets:
    """
    Initialse the CreateDataSets object.

    Returns
    -------
    CreateDataSets
        A CreateDataSet object.
    """
    return CreateDataSets()

def get_list_of_tifs_from_folder(create_datasets: CreateDataSets):

    test_folder_path = path.join(path.dirname(path.path.dirname(__file__)), "test_inputs", "rasters")

    expected_result_1 = ["4x4.tif_1", "4x4_2.tif", "4x4_3.tif"]
    expected_result_2 = ["4x4.tif_1", "4x4_2.tif", "4x4_3.tif", "4x4_3.jpg"]
    assert create_datasets.get_list_of_tiffs_from_folder(folder_path=test_folder_path) == expected_result_1
    assert create_datasets.get_list_of_tiffs_from_folder(folder_path=test_folder_path) != expected_result_2
