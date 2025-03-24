"""Module for reading and writing GeoZarr files."""

import json
import os
import os.path as path
from typing import Any, List, Union

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
        storage_type : str
            Define the storage type used, currently can be one of 'memory_store' or 'local_store'.
        storage_directory : str, optional
            The storage directory if the 'local_store' optins is used, by default None.
        zarr_format : int, optional
            The zarr specification used, note this can be 2 or 3, but GeoZarr is based on Zarr 2, by default 2.
        n_chunks : int, optional
            The number of chunks that the data will be chunked into a long each dimension. The actual number of
            chunks for an array of n_dims will be n_chunks^n_dims.

        Raises
        ------
        TypeError
            Raised when no storage directory is specified when the 'local_store' parameter is passed.
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
        !!!!!!!!!
        """
        data = {"standard_name": ["time", "grid_latitude", "grid_longitude", "elevation", "DEMdiff", "error"],
                "axis": ["T", "Y", "X", "None", "None", "None"],
                "bounds": ["time_bnds", "None", "None", "None", "None", "None"],
                "calendar": ["standard", "None", "None", "None", "None", "None"],
                "long_name": ["time", "grid_latitude", "grid_longitude", "elevation", "DEMdiff", "error"],
                "units": ["seconds since 1970-01-01T00:00:00", "degrees", "None", "m", "m", "m"]}

        self.dimensions_config = pd.DataFrame(data)

    @staticmethod
    def _reshape_data(data: pd.DataFrame,
                      target_column:str,
                      coord_columns: List[str]) -> np.ndarray:
        """
        Reshape a 2-dimensional array into an N-dimensional array based on a specified number of coordinate columns.

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
        np.ndarray
            The reshape data array in the form of an np.ndarray.
        """
        reshape_dims = [len(data[column].unique()) for column in coord_columns]
        data_reshape = data[target_column].to_numpy().reshape(reshape_dims)

        return data_reshape

    @staticmethod
    def _reduce_coordinates(data: pd.DataFrame,
                            target_column: str) -> np.ndarray:
        """
        Reduce a coordinate variabel to a sorted monotonically increasing set of coordinates.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be processed including the target feature in the form of a pd.DataFrame.
        target_column : str
            The coordinate feature to be processed.

        Returns
        -------
        np.ndarray
            The processed coordinate variable. This will be monotonically increasing and contain no duplicates.
        """
        return np.sort(data[target_column].unique())

    def _calculate_chunk_dims(self: Any,
                              data_shape: tuple) -> tuple:
        """
        Calculate the dims of each chunk in the Zarr file. The following fomula is applied on each dim.

                  chunk_dim = ceil(n_elements / n_chunks)

        This means that when the n_elements is not divisable by the n_chunks that the final chunk will overhang
        the edge of the array. This is standard in Zarr.

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
                          auxiliary_columns: Union[List, None]) -> Union[None, zarr.Group]:
        """
        Convert a pandas dataframe to geozarr. Note this currently assumes that the column names will follow
        CF naming conventions. It also currently assumes that the gird used, i.e. the x/y grid or the lat/long grid
        will be the same at all datapoints and time points.

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

        Returns
        -------
        Union[None, zarr.Group]
            If the geozarr file is created as a memory store then it is returned else None is returned.
        """
        # Sort the column names alphabetically:
        data_columns = sorted(data_columns)
        coordinate_columns = sorted(coordinate_columns)
        auxiliary_columns = sorted(auxiliary_columns)

        # Order the columns alpabetically between the different column groups
        data = pd.concat([data[data_columns],
                          data[coordinate_columns],
                          data[auxiliary_columns]], axis=1)

        data = data.sort_values(coordinate_columns)

        self.root.attrs["_ARRAY_DIMENSIONS"] = coordinate_columns

        for column in data_columns:
            standard_dimensions = self.dimensions_config[self.dimensions_config["standard_name"]==column]
            if standard_dimensions.shape[0] > 0:
                attributes = {key: standard_dimensions[key].iloc[0] for key in standard_dimensions.columns if
                              standard_dimensions[key].iloc[0] != "None"}
                attributes["_ARRAY_DIMENSIONS"] = coordinate_columns
            else:
                raise ValueError(f"Column {column} is not a CF standard name, please check this")
            data_reshape = self._reshape_data(data=data,
                                              target_column=column,
                                              coord_columns=coordinate_columns)
            self._add_data_array(data=data_reshape,
                                 dtype=data[column].to_numpy().dtype,
                                 name=column,
                                 attributes=attributes,
                                 chunk_shape=self._calculate_chunk_dims(data_reshape.shape))

        for column in coordinate_columns:
            standard_dimensions = self.dimensions_config[self.dimensions_config["standard_name"]==column]
            if standard_dimensions.shape[0] > 0:
                attributes = {key: standard_dimensions[key].iloc[0] for key in standard_dimensions.columns}
                attributes["_ARRAY_DIMENSIONS"] = [column]
            else:
                raise ValueError(f"Column {column} is not a CF standard name, please check this")
            data_coord = np.sort(data[column].unique())
            self._add_data_array(data=data_coord,
                                 dtype=data[column].to_numpy().dtype,
                                 name=column,
                                 attributes=attributes,
                                 chunk_shape=self._calculate_chunk_dims(data_coord.shape))

        zarr.consolidate_metadata(self.store)

        if self.storage_type == "memory_store":
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
        data : np.ndarray
            The data to be added to a data_array.
        dtype : np.typing.DTypeLike
            The dtype of the array.
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
        Load the GeoZarr file into an xarray.

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
