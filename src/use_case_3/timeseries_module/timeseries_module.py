"""Module for processing data into timeseries."""

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


class TimeSeriesModule:
    """The time series module for calculating and determining statistics."""

    def __init__(self) -> None:
        """
        Needs to be able to return summary statistics with different aggregations in time. Needs to be able
        to process and combine uncertainties.
        """
        self.aggregation_types = ("mean", "median", "max", "min")
        self.uncertainty_types = ("std", "max", "min")

    @staticmethod
    def _move_axes_time(data: np.ndarray, dimension_names: List[str]) -> np.ndarray:
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
        np.ndarray
            The reordered data
        """
        if "time" in dimension_names:
            dimension_aggregate = dimension_names.index("time")
        else:
            dimension_aggregate = 0

        if dimension_aggregate != 0:
            data_reordered = np.moveaxis(data, dimension_aggregate, 0)
        else:
            data_reordered = data

        return data_reordered

    @staticmethod
    def _move_axes_time_labels(dimension_names: List[str]) -> List[str]:
        """
        Reorder the axis so that the time axis is the first axis. If time is not in dimension names, then it
        returns the same data array and associated dimension names. This only operates on the labels and not the array.

        Parameters
        ----------
        dimension_names : List[str]
            List of dimension names with indexes corresponding to the axes in data.

        Returns
        -------
        List[str]
            The reordered data and dimension names
        """
        if "time" in dimension_names:
            dimension_aggregate = dimension_names.index("time")
        else:
            dimension_aggregate = 0

        if dimension_aggregate != 0:
            dimension_names_reordered = [dimension_names.pop(dimension_aggregate)] + dimension_names
        else:
            dimension_names_reordered = dimension_names.copy()

        return dimension_names_reordered

    def _expand_dimensions(
        self: Any, data: np.ndarray, dimension_names: List[str], modify_dimension_names: bool = False
    ) -> Tuple[np.ndarray, list]:
        """
        Check the dimensions of the input array and if not 3D then add dummy dimensions to make the data 3D. The data
        array is padded with dummy dimensions, while the dimension names have the name '_DUMMY' added at the correct
        axis.

        Parameters
        ----------
        data : np.ndarray
            The input data.
        dimension_names : List[str]
            The input dimension names.
        modify_dimension_names : bool, optional
            Whether the dimension_names should be modified, this is used when multiple data arrays with the same
            dimensions need to be modified, by default False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, list]
            A 3D data array with three named dimensions.
        """
        if len(dimension_names) == 2:
            data = np.expand_dims(data, axis=0)
            if modify_dimension_names:
                dimension_names = ["blank"] + dimension_names
        elif len(dimension_names) == 1:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)
            if modify_dimension_names:
                dimension_names = ["blank"] * 2 + dimension_names

        return data, dimension_names

    def find_time_indexes(self: Any, time_coord: pd.Series, period: str) -> dict:
        time_coord = time_coord.to_frame().reset_index(drop=True)
        time_coord["time"] = pd.to_datetime(time_coord["time"], format="%Y/%m/%d")
        time_coord["year"] = time_coord["time"].dt.year
        time_coord["date_month"] = pd.to_datetime(
            [f"{y}-{m}-01" for y, m in zip(time_coord["year"], time_coord["time"].dt.month)]
        )

        if period == "year":
            return {year: time_coord[time_coord["year"] == year].index for year in time_coord["year"].unique()}
        elif period == "month":
            return {
                date_month: time_coord[time_coord["date_month"] == date_month].index
                for date_month in time_coord["date_month"].unique()
            }

    def process_time_indexes(self: Any, dimension_data: Union[xr.DataArray], period: Union[None, str]) -> dict:
        if period is not None:
            return self.find_time_indexes(time_coord=dimension_data["time"].to_pandas(), period=period)
        else:
            return {datetime: [idx] for idx, datetime in enumerate(dimension_data["time"].values)}

    @staticmethod
    def spread_calculation_type(aggregate: str) -> Tuple[Union[str, None], bool]:
        if aggregate == "mean":
            aggregate_spread = "std"
            count_n = True
        elif aggregate == "median":
            aggregate_spread = "IQR"
            count_n = True
        elif aggregate in ["max", "min"]:
            aggregate_spread = None
            count_n = False

        return aggregate_spread, count_n

    def calculate_aggregate_variable(
        self: Any,
        data_observations: np.ndarray,
        dimension_data: dict,
        dimension_names: List[str],
        aggregate: str = "mean",
        period: Union[None, str] = None,
    ) -> pd.DataFrame:
        """
        Aggregate 2D or 3D data variable into a 1D time series using a specified aggregation. A 3D
        dataset will result in a 1D timeseries, while a 2D dataset will result in a single point in time.

        Parameters
        ----------
        data_observations : np.ndarray
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

        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=True
        )
        # data_uncertainties, dimension_names = self._expand_dimensions(data=data_uncertainties, dimension_names=dimension_names, modify_dimension_names=True)

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        # data_uncertainties = self._move_axes_time(data=data_uncertainties, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        aggregate_spread, count_n = self.spread_calculation_type(aggregate=aggregate)

        time_idx = self.process_time_indexes(dimension_data=dimension_data, period=period)

        if period is not None:
            time_idx = self.find_time_indexes(time_coord=dimension_data["time"].to_pandas(), period=period)
        else:
            time_idx = {datetime: [idx] for idx, datetime in enumerate(dimension_data["time"].values)}

        data_agg, time_agg = self.aggregate_statistic(
            data=data_observations, dimension_names=dimension_names, aggregate=aggregate, time_index=time_idx
        )

        if aggregate_spread is not None:
            data_agg_spread, _ = self.aggregate_statistic(
                data=data_observations, dimension_names=dimension_names, aggregate=aggregate_spread, time_index=time_idx
            )
        else:
            data_agg_spread = np.zeros(data_agg.shape[0])

        if count_n:
            data_agg_n = self.count_nan_elements(
                data=data_observations, dimension_count=0, non_nan=True, time_index=time_idx
            )
        else:
            data_agg_n = np.zeros(data_agg.shape[0])

        return pd.DataFrame(
            np.hstack((data_agg.reshape((-1, 1)), data_agg_spread.reshape((-1, 1)), data_agg_n.reshape((-1, 1)))),
            columns=[aggregate, aggregate_spread, "n"],
            index=time_agg,
        )

    def propagate_uncertainties(
        self: Any,
        data_observations: np.ndarray,
        data_uncertainties: np.ndarray,
        dimension_data: dict,
        dimension_names: List[str],
        aggregate: str = "mean",
        uncertainty: str = "std",
        period: Union[None, str] = None,
    ) -> pd.DataFrame:
        """
        Propagate the uncertainties. A 3D dataset will result in a 1D timeseries, while a 2D dataset will
        result in a single point in time.

        Parameters
        ----------
        data_observations : np.ndarray
            Input observation data with axes ordered according to the dimension names.
        data_uncertainties : np.ndarray
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

        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=False
        )
        data_uncertainties, dimension_names = self._expand_dimensions(
            data=data_uncertainties, dimension_names=dimension_names, modify_dimension_names=True
        )

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        data_uncertainties = self._move_axes_time(data=data_uncertainties, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        time_idx = self.process_time_indexes(dimension_data=dimension_data, period=period)

        if aggregate == "mean":
            data_uncertainties_agg, time_agg = self.aggregate_uncertainties_mean(
                data_uncertainties=data_uncertainties,
                uncertainty=uncertainty,
                dimension_names=dimension_names,
                dimension_aggregate=None,
                time_index=time_idx,
            )
        elif aggregate == "max" or aggregate == "min":
            data_uncertainties_agg, time_agg = self.aggregate_uncertainties_minmax(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                uncertainty=uncertainty,
                time_index=time_idx,
            )

        data_agg_n = self.count_nan_elements(
            data=data_uncertainties, dimension_count=0, non_nan=True, time_index=time_idx
        )

        data_agg_spread, _ = self.aggregate_statistic(
            data=data_uncertainties, dimension_names=dimension_names, aggregate="std", time_index=time_idx
        )

        return pd.DataFrame(
            np.hstack(
                (data_uncertainties_agg.reshape((-1, 1)), data_agg_spread.reshape((-1, 1)), data_agg_n.reshape((-1, 1)))
            ),
            columns=["uncertainty", "std", "n"],
            index=time_agg,
        )

    @staticmethod
    def count_nan_elements(
        data: np.ndarray, dimension_count: int = 0, non_nan: bool = False, time_index: dict = None
    ) -> np.ndarray:
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
        data_agg_n = []
        for time_point in time_index.keys():
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            if non_nan:
                data_agg_n.append(np.count_nonzero(~np.isnan(data[time_min:time_max, :, :])))
            else:
                data_agg_n.append(np.count_nonzero(np.isnan(data[time_min:time_max, :, :])))
        return np.array(data_agg_n)

    @staticmethod
    def aggregate_statistic(
        data: np.ndarray,
        aggregate: str,
        dimension_names: Union[None, List[str]] = None,
        dimension_aggregate: Union[None, int, Tuple[int, int]] = None,
        time_index: Union[None, dict] = None,
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

        time_coords_agg = []
        data_agg = []

        for time_point in time_index.keys():
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            if aggregate == "mean":
                data_agg.append(np.nanmean(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == "median":
                data_agg.append(np.nanmedian(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == "max":
                data_agg.append(np.nanmax(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == "min":
                data_agg.append(np.nanmin(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == "std":
                data_agg.append(np.nanstd(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == "IQR":
                data_agg.append(np.subtract(*np.nanpercentile(data[time_min:time_max, :, :], q=[75, 25])))
                time_coords_agg.append(time_point)

        return np.array(data_agg), time_coords_agg

    def aggregate_uncertainties_mean(
        self: Any,
        data_uncertainties: np.ndarray,
        uncertainty: str,
        dimension_names: Union[None, List[str]] = None,
        dimension_aggregate: Union[None, int, Tuple[int, int]] = None,
        time_index: Union[None, dict] = None,
    ) -> np.ndarray:
        """
        Aggregate the uncertainties when aggregating using the mean. In this case the standard deviations
        are combined through quadrature.

        Parameters
        ----------
        data_uncertainties : np.ndarray
            Input data uncertainties with axes ordered according to the dimension names.
        uncertainty : str
            The type of uncertainty, currently only std is supported.
        dimension_names : Union[None, List[str]], optional
            List of dimension names with indexes corresponding to the axes in data., by default None.
        dimension_aggregate : Union[None, int, Tuple[int]], optional
            The dimension of the data array to aggregate against, by default None.

        Returns
        -------
        np.ndarray
            The aggregated uncertainties.
        """
        data_agg_n = self.count_nan_elements(
            data=data_uncertainties, dimension_count=0, non_nan=True, time_index=time_index
        )

        time_coords_agg = []
        data_agg = []

        for i, time_point in enumerate(time_index.keys()):
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            if uncertainty == "std":
                data_agg.append(
                    np.sqrt(np.nansum(np.square(data_uncertainties[time_min:time_max, :, :]))) / data_agg_n[i]
                )

        return np.array(data_agg), time_coords_agg

    def aggregate_uncertainties_minmax(
        self: Any,
        data_observations: np.ndarray,
        data_uncertainties: np.ndarray,
        uncertainty: str,
        time_index: Union[None, dict] = None,
    ) -> np.ndarray:
        """
        Aggregate the uncertainties when aggregating using either the maximum or minimum value. In this case the
        standard deviation corresponds to the pixel with either the maximum or minimum value respectively.

        Parameters
        ----------
        data_observations : np.ndarray
            Input data with axes ordered according to the dimension names.
        data_uncertainties : nd_array
            Input data uncertainties with axes ordered according to the dimension names.
        uncertainty : str
            The type of uncertainty, currently only std is supported.
        dimension_names : Union[None, List[str]], optional
            List of dimension names with indexes corresponding to the axes in data., by default None.
        dimension_aggregate : Union[None, int, Tuple[int]], optional
            The dimension of the data array to aggregate against, by default None.

        Returns
        -------
        np.ndarray
            The aggregated uncertainties.
        """
        time_coords_agg = []
        data_agg = []

        for time_point in time_index.keys():
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            data_subcube_observations = data_observations[time_min:time_max, :, :]
            data_subcube_uncertainties = data_uncertainties[time_min:time_max, :, :]
            if uncertainty == "max":
                idx = np.unravel_index(
                    np.nanargmax(data_subcube_observations, axis=None), data_subcube_observations.shape
                )
                data_agg.append(data_subcube_uncertainties[idx])
            elif uncertainty == "min":
                idx = np.unravel_index(
                    np.nanargmin(data_subcube_observations, axis=None), data_subcube_observations.shape
                )
                data_agg.append(data_subcube_uncertainties[idx])

        return np.array(data_agg), time_coords_agg
