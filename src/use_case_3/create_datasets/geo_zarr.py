"""Module for reading and writing GeoZarr files."""

from typing import Any, List, Union

import numpy as np
import pandas as pd
import zarr
import zarr.storage


class GeoZarr:
    """The GeoZarr data object. Allows the reading and writing of GeoZarr files which conform to the GeoZarr spec."""

    def __init__(self: Any,
                 storage_type: str,
                 storage_directory: str=None,
                 zarr_format: int=2):
        """
        Initialise the GeoZarr object.

        Parameters
        ----------
        self : Any
            The class.
        storage_type : str
            Define the storage type used, currently can be one of 'memory_store' or 'local_store'.
        storage_directory : str, optional
            The storage directory if the 'local_store' optins is used, by default None
        zarr_format : int, optional
            The zarr specification usid, note this can be 2 or 3, but GeoZarr is based on Zarr 2, by default 2.

        Raises
        ------
        TypeError
            Raised when no storage directory is specified when the 'local_store' parameter is passed.
        """
        self.zarr_format = zarr_format
        if storage_type == "memory_store":
            self.store = zarr.storage.MemoryStore()
        elif storage_type == "local_store":
            if storage_directory is None:
                raise TypeError("Enter a valid storage location")
            self.store = zarr.storage.LocalStore(root=storage_directory)

        self.zarr_format = zarr_format

        #This is where the zarr file is created
        self.root = zarr.create_group(store=self.store)

    def dataframe2geozarr(self,
                          data: pd.DataFrame,
                          data_columns: List,
                          coordinate_columns: List,
                          auxiliary_columns: Union[List, None]):
        """
        Convert a pandas dataframe to geozarr.

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
        """
        pass

    def _add_data_array(self,
                        data: np.ndarray,
                        dtype: np.typing.DTypeLike):
        """
        Add a data_array to the GeoZarr.

        Parameters
        ----------
        data : np.ndarray
            The data to be added to a data_array.
        dtype : np.typing.DTypeLike
            The dtype of the array.
        """
        pass
