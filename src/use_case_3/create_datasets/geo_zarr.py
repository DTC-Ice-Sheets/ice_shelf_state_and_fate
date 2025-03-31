"""Module for reading and writing GeoZarr files."""

import json
import os
import os.path as path
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import zarr.storage

from use_case_3.create_datasets.create_dataset_helper_functions import ZarrStorage


class GeoZarr:
    """The GeoZarr data object. Allows the reading and writing of GeoZarr files which conform to the GeoZarr spec."""

    def __init__(self: Any,
                 storage_type: ZarrStorage,
                 storage_directory: Union[str, None]=None,
                 zarr_format: int=2,
                 n_chunks: int=1) -> None:
        """
        Initialise the GeoZarr object.

        Parameters
        ----------
        storage_type : ZarrStorage
            Define the storage type used, currently can be one of ZarrStorage.memory_store or ZarrStorage.local_store.
        storage_directory : str, optional
            The storage directory if the ZarrStorage.local_store option is used, by default None.
        zarr_format : int, optional
            The zarr specification used, note this can be 2 or 3, but GeoZarr is based on Zarr 2, by default 2.
        n_chunks : int, optional
            The number of chunks that the data will be chunked into a long each dimension. The actual number of
            chunks for an array of n_dims will be n_chunks^n_dims.

        Raises
        ------
        TypeError
            Raised when no storage directory is specified when the 'ZarrSorage.local_store' parameter is passed.
        """
        self.zarr_format = zarr_format
        self.n_chunks = n_chunks
        self.storage_type = storage_type
        self.storage_directory = storage_directory
        if self.storage_type == ZarrStorage.memory_store:
            self.store = zarr.storage.MemoryStore()
        elif self.storage_type == ZarrStorage.local_store:
            if self.storage_directory is None:
                raise TypeError("Enter a valid storage location")
            self.store = zarr.storage.LocalStore(root=self.storage_directory)

        #This is where the zarr file is created according to the storage type chosen
        self.root = zarr.create_group(store=self.store,
                                      zarr_format=self.zarr_format)
        self.create_dimensions_config()

    def create_dimensions_config(self) -> None:
        """
        Create the dimensions config file. This is used when creating the GeoZarr file to populate the
        the metadata correctly.

        !!!!!!!!!
        Note this is currently a place holder and some of these do not follow CF standard naming conventions.
        We think that this will probably be some kind of external look-up dictionary held outside of the class
        where we can populate this for all of the expected standard names.
        !!!!!!!!!
        """
        data = {"standard_name": ["time",
                                  "grid_latitude",
                                  "grid_longitude",
                                  "height_above_mean_sea_level",
                                  "height_difference_from_dem",
                                  "height_above_mean_sea_level_uncertainty",
                                  "height_difference_from_dem_uncertainty",
                                  "basal_melt_rate",
                                  "projection_x_coordinate",
                                  "projection_y_coordinate"],
                "axis": ["T",
                         "Y",
                         "X",
                         "None",
                         "None",
                         "None",
                         "None",
                         "None",
                         "X",
                         "Y"],
                "bounds": ["time_bnds",
                           "None",
                           "None",
                           "None",
                           "None",
                           "None",
                           "None",
                           "None",
                           "None",
                           "None"],
                "calendar": ["standard",
                             "None",
                             "None",
                             "None",
                             "None",
                             "None",
                             "None",
                             "None",
                             "None",
                             "None"],
                "long_name": ["time",
                              "grid_latitude",
                              "grid_longitude",
                              "height_above_mean_sea_level",
                              "height_difference_from_dem",
                              "height_above_mean_sea_level_uncertainty",
                              "height_difference_from_dem_uncertainty",
                              "basel_melt",
                              "projection_x_coordinate",
                              "projection_y_coordinate"],
                "units": ["seconds since 1970-01-01T00:00:00",
                          "degrees",
                          "degrees",
                          "m",
                          "m",
                          "m",
                          "m",
                          "m/s",
                          "m",
                          "m"]}

        self.dimensions_config = pd.DataFrame(data)

    @staticmethod
    def _reshape_data(data: pd.DataFrame,
                      target_column:str,
                      coord_columns: List[str]) -> Tuple[np.ndarray, list[np.ndarray]]:
        """
        Reshape a 2-dimensional array into an N-dimensional array based on a specified number of coordinate columns.
        This ensures a uniformly spaced grid in the coordinates. In cases where the coordinate period is not uniform,
        this is resampled to the highest common factor. Note, resampling only works with coordinates which are integers.

        TODO: This needs optimised, there are to many for loops and some of this can probably be vectorised.
        TODO: This only tested for 2D --> 3D data transformations.

        Parameters
        ----------
        data : pd.DataFrame
            The data to reshape in the form of a pd.DataFrame.
        target_column : str
            The targe feature which will be reshaped.
        coord_columns : List[str]
            The coordinate columns indexing the target feature.

        Returns
        -------
        Tuple[np.ndarray, list[np.ndarray]
            The reshaped data array in the form of an np.ndarray.
            The coordinates of the reshaped data array in the form of an np.ndarray.
        """
        # Get a list of data vectors for each coordinate.
        coords_list = []
        for coord_column in coord_columns:
            coords_list.append(data[coord_column].values)

        # Get the vector of target data.
        target = data[target_column].values

        # Create empty lists to hold important information regardin coordinate remapping.
        coords_list_unique = [] # This holds the unique coordinate values for each coordinate variable.
        coords_idx = [] # This holds the coordinate indexes for each coordinate variable.
        data_coords = [] # used to store the final coordinate vectors which are resampled.
        for i, coord in enumerate(coords_list):
            # This calculates the difference between the coordinates for each variable, if it is not spaced evenly,
            # then this is resampled. This assumes that the data to be resampled is an integer!
            if not np.all(np.diff(np.unique(np.sort(coord)))==np.diff(np.unique(np.sort(coord)))[0]):
                period = np.min(np.gcd(np.diff(np.unique(np.sort(coord))).astype(int),
                                    np.diff(np.unique(np.sort(coord)))[0].astype(int)))
            else:
                period = np.diff(np.unique(np.sort(coord)))[0]
            # Create a sorted list of unique coordinate values.
            coords_list_unique.append(np.arange(np.min(coord), np.max(coord)+period, period))
            coords_idx.append(np.zeros(coords_list[i].shape))
            data_coords.append(coords_list_unique[i])

        # Map the orignal coordinates to the index of the new coordinate mapping, i.e. from the long form 2d data to the
        # N-dim tensor.
        for i in range(len(coords_list)):
            coords_idx[i] = coords_list_unique[i].searchsorted(coords_list[i])

        # Get the expected shape of the N-dim tensor.
        shape_matrix = [n.shape[0] for n in coords_list_unique]

        # Create the N-dim tensor and fill with NaN values.
        data_reshape = np.empty(shape_matrix)
        data_reshape[:] = np.nan

        # Create an array of coordinates where the N-Dim tensor will be filled with the data.
        final_idx = np.hstack([x.reshape((x.shape[0], -1)).astype(int) for x in coords_idx])

        # Iterate through the coordinates and set the data in the correct positions in the N-Dim array.
        # This should be possible to do through one assignment operation, but I at the moment I can't do this with np!
        for i in range(final_idx.shape[0]):
            idx = tuple([final_idx[i, j] for j in range(final_idx.shape[1])])
            data_reshape[idx] = target[i]

        return data_reshape, data_coords

    def _calculate_chunk_dims(self: Any,
                              data_shape: tuple) -> tuple:
        """
        Calculate the dims of each chunk in the Zarr file. The following fomula is applied on each dim.

                  chunk_dim = ceil(n_elements / n_chunks)

        When n_elements is not exactly divisable by n_chunks, the final chunk will overhang the edge of the array.
        This is standard in Zarr. i.e for the case of a 3x3 array with n_chunks = 2 will result in 4 2x2 chunks:

                            col1 col2  col3
                          |----------|------------|
                        0 |  a    d  |  g    NaN  |
                          |          |            |
                        1 |  b    e  |  h    NaN  |
                        --|----------|------------|
                        2 |  c    f  |  i    NaN  |
                          |          |            |
                          | NaN  NaN | NaN   NaN  |
                          |----------|------------|

        Parameters
        ----------
        data_shape : tuple
            The shape of the array being chunked.

        Returns
        -------
        tuple
            The dimensions of each chunk.
        """
        chunk_dims = [int(np.ceil(x / self.n_chunks)) for x in data_shape]

        return tuple(chunk_dims)


    def dataframe2geozarr(self,
                          data: pd.DataFrame,
                          data_columns: List,
                          coordinate_columns: List,
                          auxiliary_columns: Union[List, None],
                          map_projection: str) -> Union[None, zarr.Group]:
        """
        Convert a pandas dataframe to geozarr. Note this currently assumes that the column names will follow
        CF naming conventions.

        TODO: Auxiliary data is currently not handled.

        Parameters
        ----------
        data : pd.DataFrame
            The Pandas DataFrame object to convert to the GeoZarr object.
        data_columns : List
            The data column names which specify the Zarr DataArrays.
        coordinate_columns : List
            The column names specifiying the coordinate columns.
        auxiliary_columns : Union[List, None]
            The columns names specifying any Auxiliary columns to be included.
        map_projection : str
            The projection code used.

        Returns
        -------
        Union[None, zarr.Group]
            If the geozarr file is created as a memory store then it is returned else None is returned.

        Raises
        ------
        ValueError
            Raised when column names are not in theself.dimensions_config dict.
        """
        # Sort the column names alphabetically:
        data_columns = sorted(data_columns)
        coordinate_columns = sorted(coordinate_columns)
        auxiliary_columns = sorted(auxiliary_columns)

        # Order the columns alpabetically between the different column groups
        data = pd.concat([data[data_columns],
                          data[coordinate_columns],
                          data[auxiliary_columns]], axis=1)

        # Sort the data by the coordinate columns
        data = data.sort_values(coordinate_columns)

        # Create the necessary GeoZarr attributes for the coordinate columns.
        self.root.attrs["_ARRAY_DIMENSIONS"] = coordinate_columns
        self.root.attrs["MAP_PROJECTION"] = map_projection

        # Create the N-Dim array for each sepcified data column - According to GeoZarr this should be geospacial
        # observations.
        for column in data_columns:
            standard_dimensions = self.dimensions_config[self.dimensions_config["standard_name"]==column]
            if standard_dimensions.shape[0] > 0:
                attributes = {key: standard_dimensions[key].iloc[0] for key in standard_dimensions.columns if
                              standard_dimensions[key].iloc[0] != "None"}
                attributes["_ARRAY_DIMENSIONS"] = coordinate_columns
            else:
                raise ValueError(f"Column {column} is not a CF standard name, please check this")

            data_reshape, data_reshape_coords = self._reshape_data(data=data,
                                                                   target_column=column,
                                                                   coord_columns=coordinate_columns)

            self._add_data_array(data=data_reshape,
                                 dtype=data[column].to_numpy().dtype,
                                 name=column,
                                 attributes=attributes,
                                 chunk_shape=self._calculate_chunk_dims(data_reshape.shape))

        # Process the coordinate coluumns
        for i, column in enumerate(coordinate_columns):
            standard_dimensions = self.dimensions_config[self.dimensions_config["standard_name"]==column]
            if standard_dimensions.shape[0] > 0:
                attributes = {key: standard_dimensions[key].iloc[0] for key in standard_dimensions.columns}
                attributes["_ARRAY_DIMENSIONS"] = [column]
            else:
                raise ValueError(f"Column {column} is not a CF standard name, please check this")
            data_coord = data_reshape_coords[i]
            self._add_data_array(data=data_coord,
                                 dtype=data[column].to_numpy().dtype,
                                 name=column,
                                 attributes=attributes,
                                 chunk_shape=self._calculate_chunk_dims(data_coord.shape))

        # Encouraged by XArray to increase the speed at which the Zarr file can be loaded. This consolidates all
        # metadata to the root metadata file.
        zarr.consolidate_metadata(self.store)

        # If the storage type chosen is memory only then return the Zarr object, else the method returns nothing.
        if self.storage_type == ZarrStorage.memory_store:
            return self.root

    def _add_data_array(self,
                        attributes: dict[str, zarr.core.common.JSON],
                        data: np.ndarray,
                        dtype: np.typing.DTypeLike,
                        name: str,
                        chunk_shape: tuple) -> None:
        """
        Add a data_array to the GeoZarr.

        Parameters
        ----------
        attributes : dict[str, zarr.core.common.JSON]
            The attributes to be written into the attribute file for the variable.
        data : np.ndarray
            The data to be added to a data_array.
        dtype : np.typing.DTypeLike
            The dtype of the array.
        name : str
            The name of the variable.
        chunk_shape : tuple
            The shape of a single chunk.
        """
        self.root.create_array(name=name,
                               shape=data.shape,
                               attributes=json.loads(json.dumps(attributes)),
                               dtype=dtype,
                               chunks=chunk_shape)

        data_slice = []
        for dim in data.shape:
            data_slice.append(slice(0, dim))

        self.root[name].set_basic_selection(tuple(data_slice), data)

    @staticmethod
    def load_geo_zarr(file_path: str) -> dict:
        """
        Load a GeoZarr file save saved on disk as a local store. This serialises
        the data and return a dict like object, where the keys are the array names
        and the values are np.ndarrays.

        The aim was for this to be a thin wrapper around zarr.load(), however, this currently
        throws: 'NotImplementedError: loading groups not yet supported'. Instead, this has
        been implemented by scanning the zarr file root directory and loading each array into
        memory. Currently this will only load arrays directly in the root zarr folder.

        There are other options for loading this zarr file without loading everything into
        memory such as using the self.load_geo_zarr_xarray, which is the recommended method.

        TODO: Recursively load non root data (this can only be auxiliary data according to
              the GeoZarr specification).

        Parameters
        ----------
        file_path : str
            The file path to the root zarr folder group location.

        Returns
        -------
        dict
            The loaded GeoZarr file as a dict-like object
        """
        data_arrays = os.listdir(file_path)

        data_arrays = [x for x in data_arrays if x[0] != "."]

        zarr_dict = {feature: zarr.load(path.join(file_path, feature)) for feature in data_arrays \
          if ".zarray" in os.listdir(path.join(file_path, feature))}

        return zarr_dict

    def load_geo_zarr_xarray(self: Any,
                             file_path: str) -> xr.DataArray:
        """
        Load the GeoZarr file into an xarray. Thin wrapper around xarray.open_zarr.

        Parameters
        ----------
        file_path : str
            The file path to the root zarr folder group location.

        Returns
        -------
        xarray
            The xarray of the zarr file
        """
        return xr.open_zarr(store=file_path,
                            zarr_format=self.zarr_format)
