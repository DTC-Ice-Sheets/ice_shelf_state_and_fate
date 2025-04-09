"""Module for processing data into timeseries."""

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd


class TimeSeriesModule:
    """The time series module for calculating and determining statistics."""

    def __init__(self) -> None:
        """
        Needs to be able to return summary statistics with different aggregations in time. Needs to be able
        to process and combine uncertainties.
        """
        self.aggregation_types = ("mean", "median", "max", "min")
        self.uncertainty_types = "std"

    @staticmethod
    def _move_axes_time(data: np.ndarray, dimension_names: List[str]) -> Tuple[np.ndarray, list]:
        """
        Reorder the axis so that the time axis is the first axis. If time is not in dimension names, then it
        returns the same data array and associated dimension names.

        Parameters
        ----------
        data : np.ndarray
            Input data with axes ordered according to the dimension names.
        dimension_names : List[str]
            List of dimension names with indexes corresponding to the axes in data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The reordered data and dimension names
        """
        if "time" in dimension_names:
            dimension_aggregate = dimension_names.index("time")
        else:
            dimension_aggregate = 0

        if dimension_aggregate != 0:
            dimension_names_reordered = [dimension_names.pop(dimension_aggregate)] + dimension_names
            data_reordered = np.moveaxis(data, dimension_aggregate, 0)
        else:
            dimension_names_reordered = dimension_names.copy()
            data_reordered = data

        return data_reordered, dimension_names_reordered

    def calculate_aggregate_variable(
        self: Any, data: np.ndarray, dimension_data: dict, dimension_names: List[str], aggregate: str = "mean"
    ) -> pd.DataFrame:
        """
        Aggregate 2D or 3D data variable into a 1D time series using a specified aggregation. A 3D
        dataset will result in a 1D timeseries, while a 2D dataset will result in a single point in time.

        Parameters
        ----------
        data : np.ndarray
            Input data with axes ordered according to the dimension names.
        dimension_data : dict
            The data for each of the data dimensions, i.e. coordinate values.
        dimension_names : List[str]
            List of dimension names with indexes corresponding to the axes in data.
        aggregate : str, optional
            The aggregation to perform on the data, by default "mean". Currently these are limited to:
            mean, median, max, min. Note, the aggregation chosen determines the spread according to the following rules:
                - mean: standard deviation
                - median: IQR
                - max/min: None

        Returns
        -------
        pd.DataFrame
            A dataframe containing the aggregated variable, the spread of that variable and the number of
            points in the aggregation. The index of the dataframe is time.

        Raises
        ------
        ValueError
            The aggregation type is not one of the specified values.
        """
        if aggregate not in self.aggregation_types:
            raise ValueError(f"The aggregation type {aggregate} is not one of {self.aggregation_types}")
        if len(dimension_names) == 3:
            data, dimension_names = self._move_axes_time(data=data, dimension_names=dimension_names)
        elif len(dimension_names) == 2:
            data = np.expand_dims(data, axis=0)
            dimension_names = ["blank"] + dimension_names
        elif len(dimension_names) == 1:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            dimension_names = ["blank"] * 2 + dimension_names

        if aggregate == "mean":
            aggregate_spread = "std"
            count_n = True
        elif aggregate == "median":
            aggregate_spread = "IQR"
            count_n = True
        elif aggregate in ["max", "min"]:
            aggregate_spread = None
            count_n = False

        data_agg = self.aggregate_statistic(data=data, dimension_names=dimension_names, aggregate=aggregate)
        if aggregate_spread is not None:
            data_agg_spread = self.aggregate_statistic(
                data=data, dimension_names=dimension_names, aggregate=aggregate_spread
            )
        else:
            data_agg_spread = np.zeros(data.shape[0])
        if count_n:
            data_agg_n = self.count_nan_elements(data=data, dimension_count=0, non_nan=True)
        else:
            data_agg_n = np.zeros(data.shape[0])

        return pd.DataFrame(
            np.hstack((data_agg.reshape((-1, 1)), data_agg_spread.reshape((-1, 1)), data_agg_n.reshape((-1, 1)))),
            columns=[aggregate, aggregate_spread, "n"],
            index=dimension_data["time"],
        )

    def propagate_uncertainties(
        self: Any,
        data: np.ndarray,
        dimension_data: dict,
        dimension_names: List[str],
        aggregate: str = "mean",
        uncertainty: str = "std",
    ) -> pd.DataFrame:
        """
        Propagate the uncertainties. A 3D dataset will result in a 1D timeseries, while a 2D dataset will
        result in a single point in time.

        Parameters
        ----------
        data : np.ndarray
            Input data uncertainties with axes ordered according to the dimension names.
        dimension_data : dict
            The data for each of the data dimensions, i.e. coordinate values.
        dimension_names : List[str]
            List of dimension names with indexes corresponding to the axes in data.
        aggregate : str, optional
            The type of aggregation performed on the data, by default "mean". Currently these are limited to:
            mean, median, max, min. Note, the aggregation chosen determines the spread according to the following rules:
                - mean: standard deviation
                - median: IQR
                - max/min: NaN
        uncertainty : str, optional
            The type of uncertainty provided, by default "std". Currently these are limited to:
            std.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the aggregated uncertainties, the spread of that uncertainties and the number of
            points in the aggregation. The index of the dataframe is time.

        Raises
        ------
        ValueError
            The aggregation type is not one of the specified values.
        ValueError
            The uncertainty type is not one of the specified values.
        """
        if aggregate not in self.aggregation_types:
            raise ValueError(f"The aggregation type {aggregate} is not one of {self.aggregation_types}")
        if uncertainty not in self.uncertainty_types:
            raise ValueError(f"The aggregation type {uncertainty} is not one of {self.uncertainty_types}")
        if len(dimension_names) == 3:
            data, dimension_names = self._move_axes_time(data=data, dimension_names=dimension_names)
        elif len(dimension_names) == 2:
            data = np.expand_dims(data, axis=0)
            dimension_names = ["blank"] + dimension_names
        elif len(dimension_names) == 1:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            dimension_names = ["blank"] * 2 + dimension_names

        if aggregate == "mean":
            data_uncertainties = self.aggregate_uncertainties_mean(
                data=data, uncertainty=uncertainty, dimension_names=dimension_names, dimension_aggregate=None
            )

        data_agg_n = self.count_nan_elements(data=data, dimension_count=0, non_nan=True)

        data_agg_spread = self.aggregate_statistic(data=data, dimension_names=dimension_names, aggregate="std")

        return pd.DataFrame(
            np.hstack(
                (data_uncertainties.reshape((-1, 1)), data_agg_spread.reshape((-1, 1)), data_agg_n.reshape((-1, 1)))
            ),
            columns=["uncertainty", "std", "n"],
            index=dimension_data["time"],
        )

    @staticmethod
    def count_nan_elements(data: np.ndarray, dimension_count: int = 0, non_nan: bool = False) -> np.ndarray:
        """
        Count the number of nan or non-nan elements in an array.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        dimension_count : int, optional
            The array axis to count nan values against, by default 0.
        non_nan : bool, optional
            If False then the number of NaN values are counted, if True then the number of non-NaN values are
            counted, by default False.

        Returns
        -------
        np.ndarray
            _description_
        """
        data_agg_n = np.zeros((data.shape[dimension_count]))
        for idx in range(data.shape[dimension_count]):
            if non_nan:
                data_agg_n[idx] = np.count_nonzero(~np.isnan(data[idx, :, :]))
            else:
                data_agg_n[idx] = np.count_nonzero(np.isnan(data[idx, :, :]))
        return data_agg_n

    @staticmethod
    def aggregate_statistic(
        data: np.ndarray,
        aggregate: str,
        dimension_names: Union[None, List[str]] = None,
        dimension_aggregate: Union[None, int, Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Calculate the mean along the none time axes resulting in a 1d timeseries of mean values with associated
        uncertainties.

        Parameters
        ----------
        data : np.ndarray
            The input n-dim data array. When the array is 1d along the time axis this will be returned with no
            aggregation.
        dims : list
            Specification of the dimension names.

        Returns
        -------
        np.ndarray
            The aggregated data.
        """
        # If the dimensions over which to aggregate are not specified, then infer this from the dim_names:
        if dimension_aggregate is None:
            dimension_aggregate = tuple([i for i, dim_name in enumerate(dimension_names) if dim_name != "time"])

        if aggregate == "mean":
            return np.nanmean(data, axis=dimension_aggregate)
        elif aggregate == "median":
            return np.nanmedian(data, axis=dimension_aggregate)
        elif aggregate == "max":
            return np.nanmax(data, axis=dimension_aggregate)
        elif aggregate == "min":
            return np.nanmin(data, axis=dimension_aggregate)
        elif aggregate == "std":
            return np.nanstd(data, axis=dimension_aggregate)
        elif aggregate == "IQR":
            return np.subtract(*np.nanpercentile(data, q=[75, 25], axis=dimension_aggregate))

    def aggregate_uncertainties_mean(
        self: Any,
        data: np.ndarray,
        uncertainty: str,
        dimension_names: Union[None, List[str]] = None,
        dimension_aggregate: Union[None, int, Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Aggregate the uncertainties when aggregating using the mean. In this case the standard deveiations
        are aggregated through quadrature.

        Parameters
        ----------
        data : np.ndarray
            Input data uncertainties with axes ordered according to the dimension names.
        uncertainty : str
            The type of uncertainty, currently only std is supported.
        dimension_names : Union[None, List[str]], optional
            List of dimension names with indexes corresponding to the axes in data., by default None
        dimension_aggregate : Union[None, int, Tuple[int]], optional
            The dimension of the data array to aggregate against, by default None

        Returns
        -------
        np.ndarray
            The aggregated uncertainties
        """
        # If the dimensions over which to aggregate are not specified, then infer this from the dim_names:
        if dimension_aggregate is None:
            dimension_aggregate = tuple([i for i, dim_name in enumerate(dimension_names) if dim_name != "time"])

        data_agg_n = self.count_nan_elements(data=data, dimension_count=0, non_nan=True)

        if uncertainty == "std":
            return np.sqrt(np.nansum(np.square(data), axis=dimension_aggregate)) / data_agg_n
