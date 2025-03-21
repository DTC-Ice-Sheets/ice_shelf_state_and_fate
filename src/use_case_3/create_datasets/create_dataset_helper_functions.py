"""Module containing a set of helper functions for creating datasets."""

import os
import os.path as path
from typing import Any

import pandas as pd
import xarray as xr

from use_case_3.raster_tools.raster_np import RasterNp


def get_list_of_tiffs_from_folder(folder_path: str) -> list[str]:
    """
    Return a list of filenames ending in the .tif extension. Note this doesn't check for duplicates.

    Parameters
    ----------
    folder_path: str
        The folder_path that will be scanned for .tif files.

    Returns
    -------
    list[str]
        The list of .tif files from the given folder.
    """
    files = os.listdir(folder_path)

    return [file for file in files if file[-4:]==".tif"]


def return_common_elements_from_two_lists(list_1: list,
                                            list_2: list,
                                            ascending: bool=False) -> list:
    """
    Return a list of the common elements in two lists. This is done by converting the lists to sets
    before checking the intersection and then returning a list of the common elements sorted in
    ascending order by default. Note that as this done by converting to sets so any duplicates will also
    be removed.

    Parameters
    ----------
    list_1: list
        A first list of elements.
    list_2: list
        A second list of elements.
    ascending: bool
        Flag to determine how the final list is sorted. If False then this will sort in ascending order, while
        if True this will be descending.

    Returns
    -------
    list
        The list containing the intersection between list_1 and list_2 and duplicates removed.
    """
    if ascending not in (True, False):
        raise Exception("sorting must be one of either True or False...")

    set_1 = set(list_1)
    set_2 = set(list_2)

    list_interesection = sorted(set_1.intersection(set_2), reverse= not ascending)

    return list_interesection


def convert_filename2date(file_name: str) -> str:
    """
    Convert a filename to a date string.

    Parameters
    ----------
    file_name: str
        The file name which consists of a date in YYYY-mm-dd format and a file extension such as .tif.
        Note this splits the string based on the '.' and assumes the first part of the split is the date.

    Returns
    -------
    str
        The date in the format YYYY-mm-dd.
    """
    return file_name.split(".")[0]


def load_tiffs2dataframe(file_names: list[str],
                         folder_path_elevations: str,
                         folder_path_errors: str,
                         dem_diff: bool = True) -> pd.DataFrame:
    """
    Load a set of .tif files correspond to both the elevation (DEMdiff) and associated errors. Return
    the loaded .tif files as a pd.DataFrame.

    Parameters
    ----------
    file_names: list[str]
        A list of filenames which are common to both the medians and the errors.
    folder_path_elevations: str
        folder path pointing to where the elevation data is stored.
    folder_path_errors: str
        folder path pointing to where the elevation error data is stored.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame object containing the elevation data and associated errors at every coordinate
        and timepoint. Note that currently this assumes that the x and y variables are the longitude and the latitude.
        This will need to be revisited.
    """
    # A list used to hold the df for each time point. These will be concatonated at the end.
    data_elevation_list = []

    if dem_diff:
        elevation_col_name = "DEMdiff"
    else:
        elevation_col_name = "elevation"

    # Using a dummy initial time at the moment
    initial_time = pd.Timestamp(year=1970,
                                month=1,
                                day=1)

    for file_name in file_names:
        date_time = convert_filename2date(file_name=file_name)

        file_path_elevation = path.join(folder_path_elevations, file_name)
        data_elevation = RasterNp.load_from_tiff(file_path=file_path_elevation).convert_to_dataframe()
        data_elevation["time"] = pd.Timestamp(year=int(date_time.split("-")[0]),
                                                month=int(date_time.split("-")[1]),
                                                day=int(date_time.split("-")[2]))

        data_elevation["time"] = pd.to_datetime(data_elevation["time"])
        data_elevation["time"] = (data_elevation["time"] - initial_time).dt.total_seconds()/60
        data_elevation["value"] = data_elevation["value"]
        data_elevation = data_elevation.rename({"value": elevation_col_name,
                                                "x": "grid_longitude",
                                                "y": "grid_latitude"}, axis=1)
        data_elevation = data_elevation.set_index(["grid_longitude", "grid_latitude", "time"])

        file_pathe_errors = path.join(folder_path_errors, file_name)
        data_error = RasterNp.load_from_tiff(file_path=file_pathe_errors).convert_to_dataframe()
        data_error["time"] = pd.Timestamp(year=int(date_time.split("-")[0]),
                                                month=int(date_time.split("-")[1]),
                                                day=int(date_time.split("-")[2]))
        data_error["time"] = (data_error["time"] - initial_time).dt.total_seconds()/60
        data_error = data_error.rename({"value": "error",
                                        "x": "grid_longitude",
                                        "y": "grid_latitude"}, axis=1)
        data_error = data_error.set_index(["grid_longitude", "grid_latitude", "time"])

        data_elevation = data_elevation.join(data_error, how="inner")
        data_elevation_list.append(data_elevation)

    return pd.concat(data_elevation_list, axis=0).reset_index(drop=False)


def load_netcdf(folder_path: str,
                file_name: str) -> pd.DataFrame:
    """
    Load a NetCDF file (.nc) and return a Pandas DataFrame object.

    Parameters
    ----------
    folder_path: str
        The folder path pointing to where the NetCDF file is stored.
    file_name: str
        The file_name to be loaded.

    Returns
    -------
    pd.DataFrame
        The loaded data as a Pandas DataFrame object.

    Raises
    ------
    ValueError
        If the file is not a NetCDF file, i.e. the extension is not .nc.
    """
    if file_name.split(".")[-1] != "nc":
        raise ValueError("The file being loaded is not a NetCDF file (i.e. does not have the extension.nc)")

    data = xr.open_dataset(path.join(folder_path, file_name))
    data = data.to_dataframe()
    return data
