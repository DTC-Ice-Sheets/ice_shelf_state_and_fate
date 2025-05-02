"""Module containing a set of helper functions for creating datasets."""

import os
import os.path as path
import re
from typing import Dict, List, Tuple, Union

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import xarray as xr

from use_case_3.raster_tools.raster_np import BoundingBox, RasterNp


def get_list_of_tiffs_from_folder(
    folder_path: str, prefix: Union[None, str] = None, suffix: Union[None, str] = None
) -> list[str]:
    """
    Return a list of filenames ending in the .tif extension. Note this doesn't check for duplicates.

    Parameters
    ----------
    folder_path: str
        The folder_path that will be scanned for .tif files.
    prefix: str, optional
        Whether the returned files must start with the specified prefix.
    suffix: str, optional
        Whether the returned files must end with the specified suffix. Noe that the specified suffix must include any
        file extension.

    Returns
    -------
    list[str]
        The list of .tif files from the given folder.
    """
    files = os.listdir(folder_path)

    if prefix is not None:
        files = [file for file in files if file.startswith(prefix)]

    if suffix is not None:
        files = [file for file in files if file.endswith(suffix)]

    return [file for file in files if file.endswith(".tif")]


def return_common_elements_from_two_lists(list_1: list, list_2: list, ascending: bool = False) -> list:
    """
    Return a list of the common elements in two lists. This is done by converting the lists to sets
    before checking the intersection and then returning a list of the common elements sorted in
    ascending order by default. Note that this is done by converting to sets so any duplicates will also
    be removed.

    Parameters
    ----------
    list_1: list
        A first list of elements.
    list_2: list
        A second list of elements.
    ascending: bool
        Flag to determine how the final list is sorted. If True then this will sort in ascending order, while
        if False this will be descending.

    Returns
    -------
    list
        The list containing the intersection between list_1 and list_2 and duplicates removed.

    Raises
    ------
    TypeError
        Raised if non boolean value is passed.
    """
    if not isinstance(ascending, bool):
        raise TypeError("sorting must be one of either True or False...")

    if len(list_1) > 0 and len(list_2) == 0:
        return sorted(list_1, reverse=not ascending)
    elif len(list_1) == 0 and len(list_2) > 0:
        return sorted(list_2, reverse=not ascending)
    elif len(list_1) > 0 and len(list_2) > 0:
        set_1 = set(list_1)
        set_2 = set(list_2)

        list_intersection = sorted(set_1.intersection(set_2), reverse=not ascending)

        return list_intersection
    else:
        return None


def extract_date_string_from_filename(file_name: str, date_format: str = "YYYY-mm-dd") -> str:
    """
    Convert a filename containing a date to a string.

    Parameters
    ----------
    file_name : str
        The file name which contains a date.
    date_format : str
        The format of the date expected which must follow either: "YYYY-mm-dd", "YYYY-dd-mm", "dd-mm-YYYY" or
        "mm-dd-YYYY".

    Returns
    -------
    str
        The date in the format YYYY-mm-dd.

    Raises
    ------
    ValueError
        If the wrong date format is found.
    """
    acceptable_formats = ["YYYY-mm-dd", "YYYY-dd-mm", "dd-mm-YYYY", "mm-dd-YYYY"]
    if date_format not in acceptable_formats:
        raise ValueError(f"The date format {date_format} is not one of {acceptable_formats}.")

    if date_format in acceptable_formats[:2]:
        match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", file_name)  # look for YYYY-mm-dd or YYYY-dd-mm pattern
    elif date_format in acceptable_formats[2:]:
        match = re.search(r"\b\d{2}-\d{2}-\d{4}\b", file_name)  # look for dd-mm-YYYY or mm-dd-YYYY pattern

    if match:
        return match.group(0)
    else:
        raise ValueError(f"No date of format {date_format} found in the filename {file_name}.")


def parse_dates_eolis_filename(file_name: str) -> str:
    """
    Parse the date from an EOLIS filename which is in a standard format.

    Parameters
    ----------
    file_name : str
        The file name which contains a date.

    Returns
    -------
    str
        The date in the format YYYY-mm-dd.
    """
    date = file_name.split("_")[4]

    return date[:4] + "-" + date[4:6] + "-" + date[6:]


def load_tiffs2dataframe(
    file_names: list[str],
    folder_path_observations: str,
    observation_names: Dict[str, str],
    coord_names: List[str],
    reference_time: Union[None, pd.Timestamp],
    error_file_names: Union[list[str], None] = None,
    folder_path_errors: Union[str, None] = None,
    file_path_mask: Union[str, None] = None,
    create_raster_mask: bool = False,
    geometry_name: Union[str, None] = None,
    process_datetimes: bool = False,
    process_filenames_as_dates: str = "dates",
) -> Tuple[pd.DataFrame, Union[RasterNp, None]]:
    """
    Load a set of .tif files and associated errors. Return the loaded .tif files as a centered pd.DataFrame.

    Parameters
    ----------
    file_names : list[str]
        A lit of filenames for the observations.
    folder_path_observations : str
        Folder path pointing to where the observation data is stored.
    observation_names : Dict[str, str]
        A mapping for the observation names.
    coord_names : List[str]
        The list of coordinate names.
    reference_time : Union[None, pd.Timestamp]
        A reference time which is used to convert the timestamps to a time unit in seconds.
    error_file_names : Union[list[str], None], optional
        The file names of the errors.
    folder_path_errors : Union[str, None], optional
        The folder path pointing to where the observation error data is stored.
    file_path_mask : Union[str, None], optional
        The path to where the file path is stored.
    create_raster_mask : bool, optional
        Whether a rasterised mask should be created from the shapefile.
    geometry_name : Union[str, None], optional
        Whether a specific geometry name should be used.
    process_datetimes : bool, optional
        Whether the datetimes should be processed.
    process_filenames_as_dates : str, optional
        Whether the filenames are to be interpreted as dates.

    Returns
    -------
    Tuple[pd.DataFrame, Union[RasterNp, None]]
        A Pandas DataFrame object containing the observation data and associated errors at every coordinate
        and time point. A rasterised version of the mask used.
    """
    data_observations, mask_raster = process_variable(
        file_names=file_names,
        folder_path_variable=folder_path_observations,
        variable_names=observation_names,
        coord_names=coord_names,
        reference_time=reference_time,
        file_path_mask=file_path_mask,
        create_raster_mask=create_raster_mask,
        geometry_name=geometry_name,
        process_datetimes=process_datetimes,
        process_filenames_as_dates=process_filenames_as_dates,
    )

    if folder_path_errors is not None:
        observation_names["value"] = observation_names["value"] + "_uncertainty"

        if error_file_names is None:
            error_file_names = file_names

        data_errors, _ = process_variable(
            file_names=error_file_names,
            folder_path_variable=folder_path_errors,
            variable_names=observation_names,
            coord_names=coord_names,
            reference_time=reference_time,
            file_path_mask=file_path_mask,
            create_raster_mask=create_raster_mask,
            geometry_name=geometry_name,
            process_datetimes=process_datetimes,
            process_filenames_as_dates=process_filenames_as_dates,
        )

        data_observations = pd.merge(data_observations, data_errors, on=coord_names)

    return data_observations, mask_raster


def process_variable(
    file_names: list[str],
    folder_path_variable: str,
    variable_names: Dict[str, str],
    coord_names: List[str],
    reference_time: Union[None, pd.Timestamp],
    file_path_mask: Union[str, None] = None,
    create_raster_mask: bool = False,
    geometry_name: Union[str, None] = None,
    process_datetimes: bool = False,
    process_filenames_as_dates: str = "dates",
) -> Tuple[pd.DataFrame, Union[RasterNp, None]]:
    """
    Process a given variable into a dataframe and mask.

    Parameters
    ----------
    file_names : list[str]
        A lit of filenames for the observations.
    folder_path_observations : str
        Folder path pointing to where the observation data is stored.
    variable_names : Dict[str, str]
        A mapping for the variable names.
    coord_names : List[str]
        The list of coordinate names.
    reference_time : Union[None, pd.Timestamp]
        A reference time which is used to convert the timestamps to a time unit in seconds.
    file_path_mask : Union[str, None], optional
        The path to where the file path is stored.
    create_raster_mask : bool, optional
        Whether a rasterised mask should be created from the shapefile.
    geometry_name : Union[str, None], optional
        Whether a specific geometry name should be used.
    process_datetimes : bool, optional
        Whether the datetimes should be processed.
    process_filenames_as_dates : str, optional
        Whether the filenames are to be interpreted as dates.

    Returns
    -------
    Tuple[pd.DataFrame, Union[RasterNp, None]]
        The processed data frame and mask.
    """
    # A list used to hold the df for each time point. These will be concatenated at the end.
    data_variable_list = []
    # Process each file.
    for i, file_name in enumerate(file_names):
        if process_filenames_as_dates == "dates":
            date_time = extract_date_string_from_filename(file_name=file_name, date_format="YYYY-mm-dd")
        elif process_filenames_as_dates == "eolis":
            date_time = parse_dates_eolis_filename(file_name=file_name)
        file_path_variable = path.join(folder_path_variable, file_name)
        data_variable = RasterNp.load_from_tiff(file_path=file_path_variable)

        # On the first iteration create the mask to be used as a raster.
        if i == 0 and file_path_mask is not None:
            mask = get_mask_from_shapefile(
                file_path_raster=file_path_variable, file_path_mask=file_path_mask, geometry_name=geometry_name
            )
            if create_raster_mask:
                shapes = gpd.read_file(file_path_mask)
                if geometry_name == "all":
                    shape = shapes
                else:
                    shape = shapes[shapes["Name"] == geometry_name]
                # Note that we need to remove one row and column from the raster as the process of creating a dataframe
                # centers the data and therefore loses one row and column.
                mask_raster = RasterNp.rasterise_mask_from_geometry(
                    geometries=shape,
                    y_min=data_variable.bounding_box.min_y,
                    y_max=data_variable.bounding_box.max_y - data_variable.resolution,
                    x_min=data_variable.bounding_box.min_x,
                    x_max=data_variable.bounding_box.max_x - data_variable.resolution,
                    resolution=data_variable.resolution,
                    y_affine_negative=True,
                )
        elif i == 0 and file_path_mask is None:
            mask_raster = None

        # Mask the data if there is a mask available.
        if file_path_mask is not None:
            data_variable.mask_from_raster_np(mask, include_mask_value=False)

        # Convert elevation data to a DataFrame and rename columns as necessary.
        data_variable = data_variable.convert_to_dataframe()
        if process_datetimes:
            data_variable["time"] = (pd.to_datetime(date_time) - reference_time).total_seconds()

        data_variable = data_variable.rename(variable_names, axis=1)
        data_variable = data_variable.set_index(coord_names)

        data_variable_list.append(data_variable)

    return pd.concat(data_variable_list, axis=0).reset_index(drop=False), mask_raster


def load_tiffs2dataframe_basalmelt(
    file_path_observations: str,
    file_path_errors: str,
    file_path_mask: Union[str, None] = None,
    glacier_name: Union[str, None] = None,
) -> pd.DataFrame:
    """
    Load a .tif files for the  and associated errors. Return
    the loaded .tif files as a pd.DataFrame.

    Parameters
    ----------
    file_path_observations: str
        folder path pointing to where the observations are stored.
    file_path_errors: str
        folder path pointing to where the observation error data is stored.
    file_path_mask : str
        file path pointing at where the mask file is stored
    glacier_name : Union[str, None]
        The glacier of interest.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame object containing the basal melt data and associated errors at every coordinate
        and time point. Note that currently this assumes that the x and y variables are the longitude and the latitude.
        This will need to be revisited.
    """
    data = RasterNp.load_from_tiff(file_path=file_path_observations)

    observation_col_name = "basal_melt_rate"

    # On the first iteration create the mask to be used as a raster.
    if file_path_mask is not None:
        mask = get_mask_from_shapefile(
            file_path_raster=file_path_observations, file_path_mask=file_path_mask, glacier_name=glacier_name
        )

    # Mask the data if there is a mask available.
    if file_path_mask is not None:
        data.mask_from_raster_np(mask, include_mask_value=False)

    # Convert elevation data to a DataFrame and rename columns as necessary.
    data = data.convert_to_dataframe()
    data["value"] = data["value"]
    data = data.rename(
        {"value": observation_col_name, "x": "projection_x_coordinate", "y": "projection_y_coordinate"}, axis=1
    )
    data = data.set_index(["projection_x_coordinate", "projection_y_coordinate"])

    # Process the errors
    data_error = RasterNp.load_from_tiff(file_path=file_path_errors)
    if file_path_mask is not None:
        data_error.mask_from_raster_np(mask, include_mask_value=False)

    data_error = data_error.convert_to_dataframe()
    data_error = data_error.rename(
        {
            "value": observation_col_name + "_uncertainty",
            "x": "projection_x_coordinate",
            "y": "projection_y_coordinate",
        },
        axis=1,
    )
    data_error = data_error.set_index(["projection_x_coordinate", "projection_y_coordinate"])

    # Join the elevation and error data together
    data = data.join(data_error, how="inner").reset_index(drop=False)

    return data


def load_netcdf(folder_path: str, file_name: str) -> pd.DataFrame:
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


def get_mask_from_shapefile(
    file_path_raster: str, file_path_mask: str, geometry_name: Union[List[str], str]
) -> RasterNp:
    """
    Convert a shapefile with a polygon defining a mask to a mask in a RasterNp object.
    In this case the mask is boolean with the same extent as the tif to be masked. Where a pixel
    is marked as False in the mask, it will be included, while True results in the pixel being
    set to a np.nan value.

    Parameters
    ----------
    file_path_raster : str
        File path to the raster to be masked.
    file_path_mask : str
        File path to the mask store, stored as a .gpkg. The 'Name' column specifies the
        ice sheet names and the 'geometry' column specifies the mask polygon.
    geometry_name : Union[List[str], str]
        The geometry of interest. if "all" is used then this is all geometries in the mask database supplied.

    Returns
    -------
    RasterNp
        a RasterNp object containing the mask.
    """
    shapes = gpd.read_file(file_path_mask)
    if geometry_name == "all":
        shape = shapes
    else:
        shape = shapes[shapes["Name"] == geometry_name]

    with rasterio.open(file_path_raster) as src:
        mask, _, _ = rasterio.mask.raster_geometry_mask(src, shape["geometry"], crop=False)

        bounds = (src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3])  # minx, miny, maxx, maxy
        bounding_box = BoundingBox(min_x=bounds[0], min_y=bounds[1], max_x=bounds[2], max_y=bounds[3])

    resolution = (bounds[2] - bounds[0]) / mask.shape[1]
    mask = RasterNp(input_array=mask, bounding_box=bounding_box, resolution=resolution)

    return mask


def get_gdal_geotransform(geotransform_affine: affine.Affine) -> str:
    """
    Take the Affine transformation matrix and converts it to the form expected by
    GDAL's SetGeoTransform() with the same coefficient order in a string format.

    # From https://gdal.org/en/stable/tutorials/geotransforms_tut.html
    GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
    GT(1) w-e pixel resolution / pixel width.
    GT(2) row rotation (typically zero).
    GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
    GT(4) column rotation (typically zero).
    GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).

    In a North up image the GeoTransform (GT):
        - GT(0), GT(3) position is the top left corner of the top left pixel of the raster
        - GT(1), GT(5) is the pixel size
        - GT(2), GT(4) coefficients are 0

    Parameters
    ----------
    geotransform_affine : affine.Affine
        The affine transformation matrix

    Returns
    -------
    str
        A string representation of the geotransform expected by GDAL.
    """
    return " ".join([str(coord) for coord in geotransform_affine.to_gdal()])


def convert_decimal_dates(decimal_dates: pd.Series) -> pd.Series:
    """
    Convert a series of dates as decimal year to a series of pd.Timestamp.

    Parameters
    ----------
    decimal_date : pd.Series
        The decimal year.

    Returns
    -------
    pd.Series
        The converted timestamp
    """
    fractional_years = decimal_dates.values.reshape((-1, 1))

    years = np.floor(fractional_years).astype(int).reshape((-1, 1))
    remainders = fractional_years - years

    # Vectorized start-of-year and next-year timestamps
    start_of_year = pd.to_datetime(
        pd.DataFrame(np.hstack((years, np.ones(years.shape), np.ones(years.shape))), columns=["year", "month", "day"])
    )
    start_of_next_year = pd.to_datetime(
        pd.DataFrame(
            np.hstack((years + 1, np.ones(years.shape), np.ones(years.shape))), columns=["year", "month", "day"]
        )
    )

    # Compute length of year in seconds (handles leap years correctly)
    seconds_in_year = (start_of_next_year - start_of_year).dt.total_seconds()

    # Add fraction of the year as timedelta
    datetimes = start_of_year + pd.to_timedelta(remainders.flatten() * seconds_in_year, unit="s")

    return pd.Series(datetimes)


def convert_datetime_seconds(datetimes: pd.Series, datum: pd.Timestamp) -> pd.Series:
    """
    Convert a series of pd.Timestamp's to seconds using a datum.

    Parameters
    ----------
    datetimes : pd.Series
        A series of timestamps
    datum : pd.Timestamp
        The datum timestamp to be used

    Returns
    -------
    pd.Series
        The times in seconds from the datum.
    """
    return (datetimes - datum).dt.total_seconds()
