"""Module containing a set of helper functions for creating datasets."""

import os
import re
from datetime import datetime
from typing import Tuple, Union
from os import path

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import xarray as xr
from tqdm import tqdm

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


def extract_date_string_from_filename(
    file_name: str, file_name_regex_pattern: str, date_format: str = "%Y-%m-%d"
) -> str:
    """
    Extract and parse a date string from a filename using a regex pattern and date format.

    Parameters
    ----------
    file_name : str
        The file name which contains a date.
    file_name_regex_pattern : str
        A regex pattern to extract the date portion from the filename. This should extract just a single date string.
    date_format : str
        The format of the date string extracted, using Python's datetime format codes (e.g., "%Y-%m-%d").

    Returns
    -------
    datetime
        The extracted time as a datetime object.

    Raises
    ------
    ValueError
        If no date is found or parsing fails.
    """
    matches = re.findall(file_name_regex_pattern, file_name)

    if not matches:
        raise ValueError(f"No date found in filename '{file_name}' using pattern '{file_name_regex_pattern}'.")
    if len(matches) > 1:
        raise ValueError(f"Multiple date matches found in filename '{file_name}': {matches}. Only one expected.")

    date_str = matches[0]

    try:
        date_obj = datetime.strptime(date_str, date_format)
    except ValueError as e:
        raise ValueError(f"Date parsing failed for '{date_str}' with format '{date_format}': {e}") from e

    return date_obj.strftime("%Y-%m-%d")


def load_tiff2datacube(
    file_names: list[str],
    folder_path_observations: str,
    reference_time: Union[None, pd.Timestamp],
    error_file_names: Union[list[str], None] = None,
    folder_path_errors: Union[str, None] = None,
    file_path_mask: Union[str, None] = None,
    create_raster_mask: bool = False,
    geometry_name: Union[list, gpd.GeoDataFrame, str, None] = None,
    process_datetimes: bool = False,
    file_name_regex_pattern: Union[None, str] = None,
    date_format: str = "%Y-%m-%d",
    extent: Union[dict, None] = None,
) -> Tuple[np.ndarray, Union[np.ndarray, None], dict, Union[RasterNp, None]]:
    """
    Load a set of .tif files and associated errors. Return the loaded .tif files as a datacube.

    Parameters
    ----------
    file_names : list[str]
        A list of filenames for the observations.
    folder_path_observations : str
        Folder path pointing to where the observation data is stored.
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
    file_name_regex_pattern : str
        A regex pattern to extract the date portion from the filename. This should extract just a single date string.
    date_format : str
        The format of the date string extracted, using Python's datetime format codes (e.g., "%Y-%m-%d").
    extent : Union[Dict[str, Union[int, float]], None], optional
        The extent of the bounding box which will be used to crop GeoTiff files.

    Returns
    -------
    Tuple[np.ndarray, Union[np.ndarray, None], dict, Union[RasterNp, None]]
        A datacube of the stacked tiff data. A dictionary containing the coordinate data and a rasterised version of the
        mask used.
    """
    data_observations, data_coords, mask_raster = process_variable_datacube(
        file_names=file_names,
        folder_path_variable=folder_path_observations,
        reference_time=reference_time,
        file_path_mask=file_path_mask,
        create_raster_mask=create_raster_mask,
        geometry_name=geometry_name,
        process_datetimes=process_datetimes,
        file_name_regex_pattern=file_name_regex_pattern,
        date_format=date_format,
        extent=extent,
    )

    if folder_path_errors is not None:
        if error_file_names is None:
            error_file_names = file_names

        data_errors, _, _ = process_variable_datacube(
            file_names=error_file_names,
            folder_path_variable=folder_path_errors,
            reference_time=reference_time,
            file_path_mask=file_path_mask,
            create_raster_mask=create_raster_mask,
            geometry_name=geometry_name,
            process_datetimes=process_datetimes,
            file_name_regex_pattern=file_name_regex_pattern,
            date_format=date_format,
            extent=extent,
        )
    else:
        data_errors = None

    return data_observations, data_errors, data_coords, mask_raster


def build_raster_mask(
    bounding_box: BoundingBox,
    resolution: Tuple[int, float],
    y_affine_negative: bool,
    file_path_mask: str,
    geometry_name: Union[str, list, gpd.GeoDataFrame],
) -> RasterNp:
    """
    Build a rasterised mask as a RasterNp object.

    Parameters
    ----------
    bounding_box : BoundingBox
        The bounding box of the mask.
    resolution : Tuple
        The resolution of the mask.
    file_path_mask : str
        The file path to where the mask is stored as a GeoDataFrame.
    geometry_name : Union[str, list, gpd.GeoDataFrame]
        The name of the geometry/ies to be included in the GeoDataFrame. These must be elements in the "Name" column of
        the GeoDataFrame.

    Returns
    -------
    RasterNp
        The rasterised mask.
    """
    shapes = gpd.read_file(file_path_mask)
    if geometry_name == "all":
        shape = shapes
    else:
        shape = shapes[shapes["Name"] == geometry_name]
    mask_raster = RasterNp.rasterise_mask_from_geometry(
        geometries=shape,
        y_min=bounding_box.min_y,
        y_max=bounding_box.max_y,
        x_min=bounding_box.min_x,
        x_max=bounding_box.max_x,
        resolution=resolution,
        y_affine_negative=y_affine_negative,
    )

    return mask_raster


def process_variable_datacube(  # noqa
    file_names: list[str],
    folder_path_variable: str,
    reference_time: Union[None, pd.Timestamp],
    file_path_mask: Union[str, None] = None,
    create_raster_mask: bool = False,
    geometry_name: Union[list, gpd.GeoDataFrame, str, None] = None,
    process_datetimes: bool = False,
    file_name_regex_pattern: Union[None, str] = None,
    date_format: str = "%Y-%m-%d",
    extent: Union[dict, None] = None,
) -> Tuple[np.ndarray, dict, Union[RasterNp, None]]:
    """
    Process a given variable into a data cube and mask.

    TODO: This needs refactoring as currently to complex. But not immediately obvious how to refactor.

    Parameters
    ----------
    file_names : list[str]
        A list of filenames for the observations.
    folder_path_variable : str
        Folder path pointing to where the observation data is stored.
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
    file_name_regex_pattern : str
        A regex pattern to extract the date portion from the filename. This should extract just a single date string.
    date_format : str
        The format of the date string extracted, using Python's datetime format codes (e.g., "%Y-%m-%d").
    extent : Union[Dict[str, Union[int, float]], None], optional
        The extent of the bounding box which will be used to crop GeoTiff files.

    Returns
    -------
    Tuple[np.ndarray, dict, Union[RasterNp, None]]
        The processed data cube, coordinate data and mask.
    """
    # A list used to hold the raster for each time point. These will be concatenated at the end.
    data_variable_list = []
    data_timepoint_list = []
    data_coords = {}
    # Process each file.
    for i, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        date_time = extract_date_string_from_filename(
            file_name=file_name,
            file_name_regex_pattern=file_name_regex_pattern,
            date_format=date_format,
        )
        file_path_variable = path.join(folder_path_variable, file_name)
        if extent is None:
            bounding_box = None
        else:
            bounding_box = BoundingBox(
                min_x=extent["min_x"], max_x=extent["max_x"], min_y=extent["min_y"], max_y=extent["max_y"]
            )
        data_variable = RasterNp.load_from_tiff(file_path=file_path_variable, bounding_box=bounding_box)

        # Add the coordinates to the data_coords dict
        if i == 0:
            geotransform = get_gdal_geotransform(
                data_variable._get_transform(
                    bounding_box=data_variable.bounding_box,
                    resolution=data_variable.resolution,
                    y_affine_negative=data_variable.y_affine_negative,
                )
            )

            data_coord_y, data_coord_x = calculate_coords_from_geotransform(
                geotransform=geotransform, x_dim=data_variable.shape[1], y_dim=data_variable.shape[0]
            )

            data_coords["projection_y_coordinate"] = data_coord_y
            data_coords["projection_x_coordinate"] = data_coord_x

            # On the first iteration create the mask to be used as a raster.
            if file_path_mask is not None:
                mask = get_mask_from_shapefile(
                    file_path_raster=file_path_variable, file_path_mask=file_path_mask, geometry_name=geometry_name
                )
                if create_raster_mask:
                    mask_raster = build_raster_mask(
                        bounding_box=data_variable.bounding_box,
                        resolution=data_variable.resolution,
                        y_affine_negative=data_variable.y_affine_negative,
                        file_path_mask=file_path_mask,
                        geometry_name=geometry_name,
                    )

            elif file_path_mask is None:
                mask_raster = None

        if (
            data_variable.shape[0] != data_coords["projection_y_coordinate"].shape[0]
            or data_variable.shape[1] != data_coords["projection_x_coordinate"].shape[0]
        ):
            raise ValueError("The GeoTiff do not have the same coordinates")

        # Mask the data if there is a mask available.
        if file_path_mask is not None:
            data_variable.mask_from_raster_np(mask, include_mask_value=False)

        # If the datetime is to be processed then do this now
        if process_datetimes:
            data_timepoint_list.append((pd.to_datetime(date_time) - reference_time).total_seconds())

        data_coords["time"] = np.array(data_timepoint_list)
        data_variable_list.append(np.expand_dims(data_variable, axis=0))

    return np.concat(data_variable_list, axis=0), data_coords, mask_raster


def load_tiffs2dataframe(
    file_names: list[str],
    folder_path_observations: str,
    observation_names: dict[str, str],
    coord_names: list[str],
    reference_time: Union[None, pd.Timestamp],
    error_file_names: Union[list[str], None] = None,
    folder_path_errors: Union[str, None] = None,
    file_path_mask: Union[str, None] = None,
    create_raster_mask: bool = False,
    geometry_name: Union[list, gpd.GeoDataFrame, str, None] = None,
    process_datetimes: bool = False,
    file_name_regex_pattern: Union[None, str] = None,
    date_format: str = "%Y-%m-%d",
) -> Tuple[pd.DataFrame, Union[RasterNp, None]]:
    """
    Load a set of .tif files and associated errors. Return the loaded .tif files as a centered pd.DataFrame.

    Parameters
    ----------
    file_names : list[str]
        A list of filenames for the observations.
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
    file_name_regex_pattern : str
        A regex pattern to extract the date portion from the filename. This should extract just a single date string.
    date_format : str
        The format of the date string extracted, using Python's datetime format codes (e.g., "%Y-%m-%d").

    Returns
    -------
    Tuple[pd.DataFrame, Union[RasterNp, None]]
        A Pandas DataFrame object containing the observation data and associated errors at every coordinate
        and time point. A rasterised version of the mask used.
    """
    data_observations, mask_raster = process_variable_dataframe(
        file_names=file_names,
        folder_path_variable=folder_path_observations,
        variable_names=observation_names,
        coord_names=coord_names,
        reference_time=reference_time,
        file_path_mask=file_path_mask,
        create_raster_mask=create_raster_mask,
        geometry_name=geometry_name,
        process_datetimes=process_datetimes,
        file_name_regex_pattern=file_name_regex_pattern,
        date_format=date_format,
    )

    if folder_path_errors is not None:
        observation_names["value"] = observation_names["value"] + "_uncertainty"

        if error_file_names is None:
            error_file_names = file_names

        data_errors, _ = process_variable_dataframe(
            file_names=error_file_names,
            folder_path_variable=folder_path_errors,
            variable_names=observation_names,
            coord_names=coord_names,
            reference_time=reference_time,
            file_path_mask=file_path_mask,
            create_raster_mask=create_raster_mask,
            geometry_name=geometry_name,
            process_datetimes=process_datetimes,
            file_name_regex_pattern=file_name_regex_pattern,
            date_format=date_format,
        )

        data_observations = pd.merge(data_observations, data_errors, on=coord_names)

    return data_observations, mask_raster


def process_variable_dataframe(
    file_names: list[str],
    folder_path_variable: str,
    variable_names: dict[str, str],
    coord_names: list[str],
    reference_time: Union[None, pd.Timestamp],
    file_path_mask: Union[str, None] = None,
    create_raster_mask: bool = False,
    geometry_name: Union[list, gpd.GeoDataFrame, str, None] = None,
    process_datetimes: bool = False,
    file_name_regex_pattern: Union[None, str] = None,
    date_format: str = "%Y-%m-%d",
) -> Tuple[pd.DataFrame, Union[RasterNp, None]]:
    """
    Process a given variable into a dataframe and mask.

    Parameters
    ----------
    file_names : list[str]
        A list of filenames for the observations.
    folder_path_variable : str
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
    file_name_regex_pattern : str
        A regex pattern to extract the date portion from the filename. This should extract just a single date string.
    date_format : str
        The format of the date string extracted, using Python's datetime format codes (e.g., "%Y-%m-%d").

    Returns
    -------
    Tuple[pd.DataFrame, Union[RasterNp, None]]
        The processed data frame and mask.
    """
    # A list used to hold the df for each time point. These will be concatenated at the end.
    data_variable_list = []
    # Process each file.
    for i, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        date_time = extract_date_string_from_filename(
            file_name=file_name,
            file_name_regex_pattern=file_name_regex_pattern,
            date_format=date_format,
        )
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
                    y_max=data_variable.bounding_box.max_y,
                    x_min=data_variable.bounding_box.min_x,
                    x_max=data_variable.bounding_box.max_x,
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

        # display(data_variable)

        data_variable_list.append(data_variable)

    return pd.concat(data_variable_list, axis=0).reset_index(drop=False), mask_raster


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
    file_path_raster: str, file_path_mask: str, geometry_name: Union[list[str], str]
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
        mask, _, _ = rasterio.mask.raster_geometry_mask(src, shape["geometry"], crop=False, all_touched=True)

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
    Normalise datetimes against a datum.

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


def calculate_coords_from_geotransform(geotransform: str, x_dim: int, y_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the coordinates of the grid from a GDAL geotransform.

    Parameters
    ----------
    geotransform : str
        The GDAL geotransform.
    x_dim : int
        The number of x-coordinates.
    y_dim : int
        The number of y-coordinates.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The y- and x-coordinates.
    """
    transform = geotransform.split(" ")
    transform = [float(x) for x in transform]

    if float(transform[3]) < 0:
        start_y = float(transform[3])
    else:
        start_y = float(transform[3])

    y_coord = np.arange(
        start_y,
        float(transform[3]) + (y_dim * float(transform[5])),
        float(transform[5]),
    )
    x_coord = np.arange(
        float(transform[0]),
        float(transform[0]) + (x_dim * float(transform[1])),
        float(transform[1]),
    )

    return y_coord, x_coord
