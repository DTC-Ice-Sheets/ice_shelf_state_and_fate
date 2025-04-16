"""Module for processing data into timeseries."""

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import bootstrap
from sklearn.linear_model import LinearRegression


class TimeSeriesModule:
    """The time series module for calculating and determining statistics."""

    def __init__(self) -> None:
        """
        Needs to be able to return summary statistics with different aggregations in time. Needs to be able
        to process and combine uncertainties.
        """
        self.aggregation_types = ("mean", "median", "max", "min")
        self.uncertainty_types = ("std", "max", "min", "IQR")
        self.periods = ("month", "year")
        self.normalisation_types = ("first", "linear_regression")

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
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)

        if modify_dimension_names:
            if len(dimension_names) == 2:
                dimension_names = ["blank"] + dimension_names
            elif len(dimension_names) == 1:
                dimension_names = ["blank"] * 2 + dimension_names

        return data, dimension_names

    def find_time_indexes(self: Any, time_coord: pd.Series, period: str) -> dict:
        """
        Find the time indexes to slice into the data-cube.

        Parameters
        ----------
        time_coord : pd.Series
            The time coordinates.
        period : str
            The period over which to aggregate, this can either be month or year.

        Returns
        -------
        dict
            Returns the dictionary mapping for each time slices to the data-cube index.

        Raises
        ------
        ValueError
            The period must be one of {self.periods}
        """
        if period not in self.periods:
            raise ValueError(f"The period must be one of {self.periods}")
        if isinstance(time_coord.iloc[0], str):
            time_coord = time_coord.str.replace("-", "/")
            time_coord = time_coord.str.replace("_", "/")
        time_coord_df = time_coord.to_frame().reset_index(drop=True)
        time_coord_df["time"] = pd.to_datetime(time_coord_df["time"], format="%Y/%m/%d")
        time_coord_df["year"] = time_coord_df["time"].dt.year.astype(int)
        time_coord_df["month"] = time_coord_df["time"].dt.month.astype(int)

        time_coord_df["date_month"] = pd.to_datetime(
            [f"{y}-{m}" for y, m in zip(time_coord_df["year"], time_coord_df["month"])]
        )

        if period == "year":
            return {year: time_coord_df[time_coord_df["year"] == year].index for year in time_coord_df["year"].unique()}
        elif period == "month":
            return {
                date_month: time_coord_df[time_coord_df["date_month"] == date_month].index
                for date_month in time_coord_df["date_month"].unique()
            }

    def process_time_indexes(
        self: Any,
        dimension_names: list,
        dimension_data: Union[xr.DataArray, pd.DataFrame, dict],
        period: Union[None, str],
    ) -> dict:
        """
        Process the time indexes. This returns a dictionary with the each key containing the indexes
        to be aggregated into a single point.

        Parameters
        ----------
        dimension_data : Union[xr.DataArray]
            The coordinate data, i.e. the grid coordinates and the time coordinates.
        period : Union[None, str]
            The time period over which to aggregate the data points. This can be 'month', 'year' or None. If no period
            is chosen then the dictionary will contain one time point per key.

        Returns
        -------
        dict
            The dictionary containing the indexes for the time points to be aggregated.
        """
        if "time" in dimension_names:
            if isinstance(dimension_data, xr.DataArray) and "time":
                time_coord = dimension_data["time"].to_pandas()
            elif isinstance(dimension_data, pd.DataFrame):
                time_coord = dimension_data["time"]
            elif isinstance(dimension_data, dict):
                time_coord = pd.Series(dimension_data["time"], name="time")

            if period is not None:
                return self.find_time_indexes(time_coord=time_coord, period=period)
            else:
                return {datetime: [idx] for idx, datetime in time_coord.items()}
        else:
            return {0: [0]}

    @staticmethod
    def spread_calculation_type(aggregate: str) -> Tuple[Union[str, None], bool]:
        """
        Specify parameters which are calculated for the spread based on the aggregate function
        applied.

        Parameters
        ----------
        aggregate : str
            The type of aggregation applied to calculate the central tendency.

        Returns
        -------
        Tuple[Union[str, None], bool]
            The aggregate function for the spread, and the counts.
        """
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

    def process_timeseries(
        self: Any,
        data_observations: np.ndarray,
        data_uncertainties: Union[None, np.ndarray],
        data_mask: Union[None, np.ndarray],
        dimension_data: dict,
        dimension_names: List[str],
        aggregate: str = "mean",
        period: Union[None, str] = None,
        uncertainty: Union[None, str] = None,
        independent: Union[None, bool] = None,
        t0_normalisation: Union[None, str] = None,
    ) -> pd.DataFrame:
        """
        Process the input data into a time series with associated uncertainties and coverage.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        data_uncertainties : Union[None, np.ndarray]
            The data uncertainties.
        data_mask : Union[None, np.ndarray]
            The data mask.
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
        period : Union[None, str], optional
            The time period over which to aggregate, by default None
        uncertainty : Union[None, str], optional
            The statistic type describing the uncertainty, by default None
        independent : Union[None, bool], optional
            Whether the uncertainties are independent or not, by default None
        t0_normalisation : Union[None, str], optional
            Whether to normalise the time series against time zero, by default None

        Returns
        -------
        pd.DataFrame
            The aggregated 1D time series.
        """
        data_agg = self.calculate_aggregate_timeseries(
            data_observations=data_observations,
            dimension_data=dimension_data,
            dimension_names=dimension_names,
            aggregate=aggregate,
            period=period,
            t0_normalisation=t0_normalisation,
        )

        data_coverage_agg = self.calculate_coverage_timeseries(
            data_observations=data_observations,
            data_mask=data_mask,
            dimension_data=dimension_data,
            dimension_names=dimension_names,
            period=period,
        )

        data_timeseries = pd.merge(data_agg, data_coverage_agg, left_index=True, right_index=True)

        if data_uncertainties is not None:
            data_uncertainties_agg = self.propagate_uncertainties_timeseries(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                dimension_data=dimension_data,
                dimension_names=dimension_names,
                aggregate=aggregate,
                uncertainty=uncertainty,
                period=period,
                independent=independent,
                t0_normalisation=t0_normalisation,
            )
            data_timeseries = pd.merge(data_timeseries, data_uncertainties_agg, left_index=True, right_index=True)

        return data_timeseries

    def calculate_aggregate_timeseries(
        self: Any,
        data_observations: np.ndarray,
        dimension_data: dict,
        dimension_names: List[str],
        aggregate: str = "mean",
        period: Union[None, str] = None,
        t0_normalisation: Union[None, str] = None,
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

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        aggregate_spread, count_n = self.spread_calculation_type(aggregate=aggregate)

        time_idx = self.process_time_indexes(
            dimension_names=dimension_names, dimension_data=dimension_data, period=period
        )

        data_agg, time_agg = self.aggregate_statistic(
            data=data_observations, dimension_names=dimension_names, aggregate=aggregate, time_index=time_idx
        )

        if t0_normalisation is not None:
            data_agg = self.t0_normalisation(data_observation=data_agg, normalisation_type=t0_normalisation)

        if aggregate_spread is not None:
            data_agg_spread, _ = self.aggregate_statistic(
                data=data_observations, dimension_names=dimension_names, aggregate=aggregate_spread, time_index=time_idx
            )
        else:
            data_agg_spread = np.zeros(data_agg.shape[0])

        if count_n:
            data_agg_n = self.count_nan_elements(data=data_observations, non_nan=True, time_index=time_idx)
        else:
            data_agg_n = np.zeros(data_agg.shape[0])

        return pd.DataFrame(
            np.hstack((data_agg.reshape((-1, 1)), data_agg_spread.reshape((-1, 1)), data_agg_n.reshape((-1, 1)))),
            columns=[aggregate, "data_" + aggregate_spread, aggregate + "_n"],
            index=time_agg,
        )

    def propagate_uncertainties_timeseries(
        self: Any,
        data_observations: np.ndarray,
        data_uncertainties: np.ndarray,
        dimension_data: dict,
        dimension_names: List[str],
        aggregate: str = "mean",
        uncertainty: str = "std",
        period: Union[None, str] = None,
        independent: bool = False,
        t0_normalisation: Union[None, str] = None,
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

        time_idx = self.process_time_indexes(
            dimension_names=dimension_names, dimension_data=dimension_data, period=period
        )

        if aggregate == "mean":
            data_uncertainties_agg, time_agg = self.aggregate_uncertainties_mean(
                data_uncertainties=data_uncertainties,
                uncertainty=uncertainty,
                time_index=time_idx,
                independent=independent,
            )
        elif aggregate == "median":
            data_uncertainties_agg, time_agg = self.aggregate_uncertainties_median(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                time_index=time_idx,
                n_repeats=100,
            )
        elif aggregate == "max" or aggregate == "min":
            data_uncertainties_agg, time_agg = self.aggregate_uncertainties_minmax(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                uncertainty=uncertainty,
                time_index=time_idx,
            )

        if t0_normalisation is not None:
            self.t0_normalisation_uncertainties(data_uncertainties_agg, normalisation_type=t0_normalisation)

        data_agg_n = self.count_nan_elements(data=data_uncertainties, non_nan=True, time_index=time_idx)

        data_agg_spread, _ = self.aggregate_statistic(
            data=data_uncertainties, dimension_names=dimension_names, aggregate="std", time_index=time_idx
        )

        return pd.DataFrame(
            np.hstack(
                (data_uncertainties_agg.reshape((-1, 1)), data_agg_spread.reshape((-1, 1)), data_agg_n.reshape((-1, 1)))
            ),
            columns=["uncertainty", "uncertainty_std", "uncertainty_n"],
            index=time_agg,
        )

    @staticmethod
    def count_nan_elements(data: np.ndarray, non_nan: bool = False, time_index: Union[None, dict] = None) -> np.ndarray:
        """
        Count the number of nan or non-nan elements in an array.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        non_nan : bool, optional
            If False then the number of NaN values are counted, if True then the number of non-NaN values are
            counted, by default False.

        Returns
        -------
        np.ndarray
            The count of either NaN or non-NaN values.
        """
        if time_index is None:
            time_index = {0: [0]}

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
        time_index: dict,
        dimension_names: Union[None, List[str]] = None,
        dimension_aggregate: Union[None, int, Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, list]:
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
        time_index: Union[None, dict] = None,
        independent: bool = False,
    ) -> Tuple[np.ndarray, list]:
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
        independent : bool, optional
            Whether the errors are treated as independent when propagating. If they are assumed to be
            independent then they are propagated using quatrature, if they are not independent, then the errors are
            summed as a worse case.

        Returns
        -------
        Tuple(np.ndarray, list)
            The aggregated uncertainties and a list of time indexes.
        """
        data_agg_n = self.count_nan_elements(data=data_uncertainties, non_nan=True, time_index=time_index)

        time_coords_agg = []
        data_agg = []

        for i, time_point in enumerate(time_index.keys()):
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            if uncertainty == "std":
                if independent:
                    data_agg.append(
                        np.sqrt(np.nansum(np.square(data_uncertainties[time_min:time_max, :, :]))) / data_agg_n[i]
                    )
                else:
                    data_agg.append(np.nansum(data_uncertainties[time_min:time_max, :, :]) / data_agg_n[i])

        return np.array(data_agg), time_coords_agg

    def aggregate_uncertainties_median(
        self: Any,
        data_observations: np.ndarray,
        data_uncertainties: np.ndarray,
        time_index: Union[None, dict] = None,
        n_repeats: int = 1000,
    ) -> Tuple[np.ndarray, list]:
        """
        Aggregate the uncertainties for the median assuming independence. This takes a sample of the
        original data and then uses bootstrapping to estimate the propagation of the uncertainty.

        Parameters
        ----------
        data_observations : np.ndarray
            The observation data.
        data_uncertainties : np.ndarray
            The uncertainty data.
        time_index : Union[None, dict], optional
            The time indexes over which to do the aggregations, by default None
        n_repeats : int, optional
            The number of repeats. This is both the number of samples drawn and the number of bootstrapping iterations,
            by default 1000

        Returns
        -------
        Tuple(np.ndarray, list)
            The aggregated uncertainties and a list of time indexes.
        """
        time_coords_agg = []
        data_agg = []

        for time_point in time_index.keys():
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            data_observation_time_point = data_observations[time_min:time_max, :, :].flatten()
            data_uncertainties_time_point = data_uncertainties[time_min:time_max, :, :].flatten()

            data_uncertainties_time_point = data_uncertainties_time_point[~np.isnan(data_observation_time_point)]
            data_observation_time_point = data_observation_time_point[~np.isnan(data_observation_time_point)]

            # Resample the data
            resampled_data = np.random.normal(
                loc=data_observation_time_point[0],
                scale=data_uncertainties_time_point[0],
                size=(data_uncertainties_time_point.shape[0], n_repeats),
            )

            # Use bootstrapping to estimate the uncertainties
            res = bootstrap((resampled_data,), np.median, confidence_level=0.95, n_resamples=n_repeats)
            data_agg.append(np.mean(res.standard_error))

        return np.array(data_agg), time_coords_agg

    def aggregate_uncertainties_minmax(
        self: Any,
        data_observations: np.ndarray,
        data_uncertainties: np.ndarray,
        uncertainty: str,
        time_index: Union[None, dict] = None,
    ) -> Tuple[np.ndarray, list]:
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
        Tuple(np.ndarray, list)
            The aggregated uncertainties and a list of time indexes.
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

    def t0_normalisation(
        self: Any,
        data_observation: np.ndarray,
        normalisation_type: str = "first",
    ) -> np.ndarray:
        """
        Normalise a 1D time series against a time zero (t0).

        Parameters
        ----------
        data_observation : np.ndarray
            The 1D timeseries to be normalised.
        normalisation_type : str, optional
            The normalisation type to be applied. Must be one of 'first' or 'linear_regression'. Using 'first' subtracts
            the first value from the whole time series, whereas using 'linear_regression' fits a unregularised linear
            regression model to to the whole time series and then uses a predicted t0 to normalise the dataset, by
            default "first".

        Returns
        -------
        np.ndarray
            The normalised 1D timeseries using the chosen normalisation method.

        Raises
        ------
        ValueError
            The normalisation type is not one of the allowed values.
        """
        if normalisation_type not in self.normalisation_types:
            raise ValueError(f"The normalisation type chosen is not one of {self.normalisation_types}")
        if normalisation_type == "first":
            return data_observation - data_observation[0]
        elif normalisation_type == "linear_regression":
            x = np.arange(data_observation.shape[0])
            model = LinearRegression().fit(x.reshape((-1, 1)), data_observation.reshape((-1, 1)))
            y_0 = model.predict(np.array(0).reshape((-1, 1)))
            return data_observation - y_0

    def t0_normalisation_uncertainties(
        self: Any,
        data_uncertainties: Union[None, np.ndarray],
        normalisation_type: str = "first",
    ) -> np.ndarray:
        """
        Propagate uncertainties when normalising a 1D time series against a time zero (t0).

        Parameters
        ----------
        data_uncertainties : np.ndarray
            The 1D timeseries to be normalised.
        normalisation_type : str, optional
            The normalisation type to be applied. Must be one of 'first' or 'linear_regression'. Using 'first' subtracts
            the first value from the whole time series, whereas using 'linear_regression' fits a unregularised linear
            regression model to to the whole time series and then uses a predicted t0 to normalise the dataset, by
            default "first".

        Returns
        -------
        np.ndarray
            The normalised 1D timeseries using the chosen normalisation method.

        Raises
        ------
        ValueError
            The normalisation type is not one of the allowed values.
        """
        if normalisation_type not in self.normalisation_types:
            raise ValueError(f"The normalisation type chosen is not one of {self.normalisation_types}")
        if normalisation_type == "first":
            return self.combine_uncertainties_quadrature(data_uncertainties, data_uncertainties[0])
        elif normalisation_type == "linear_regression":
            # This needs fixing as the uncertainties will not be propagated correctly when fitting the model.
            return self.combine_uncertainties_quadrature(data_uncertainties, data_uncertainties[0])

    @staticmethod
    def combine_uncertainties_quadrature(
        data_uncertainties_1: np.ndarray, data_uncertainties_2: Union[np.float64, np.ndarray]
    ) -> np.ndarray:
        """
        Combine uncertainties through quadrature.

        Parameters
        ----------
        data_uncertainties_1 : np.ndarray
            The first vector of uncertainties.
        data_uncertainties_2 : np.ndarray
            The second vector of uncertainties.

        Returns
        -------
        np.ndarray
            The combined uncertainty vector.
        """
        return np.sqrt(np.square(data_uncertainties_1) + np.square(data_uncertainties_2))

    def calculate_coverage_timeseries(
        self: Any,
        data_observations: np.ndarray,
        dimension_data: dict,
        dimension_names: List[str],
        data_mask: Union[None, np.ndarray] = None,
        period: Union[None, str] = None,
    ) -> pd.DataFrame:
        """
        Calculate the coverage for the timeseries. Coverage is calculated against a reference mask.
        If no mask is provided the coverage is calculated against the total pixels.

        Parameters
        ----------
        self : Any
            _description_
        data_observations : np.ndarray
            _description_
        dimension_data : dict
            _description_
        dimension_names : List[str]
            _description_
        data_mask : Union[None, np.ndarray], optional
            _description_, by default None
        period : Union[None, str], optional
            _description_, by default None

        Returns
        -------
        pd.DataFrame
            _description_
        """
        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=True
        )

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        time_idx = self.process_time_indexes(
            dimension_names=dimension_names, dimension_data=dimension_data, period=period
        )

        data_coverage, time_agg = self.determine_coverage(
            data_observations=data_observations, mask_raster=data_mask, time_index=time_idx
        )

        data_coverage_n = self.count_nan_elements(data=data_observations, non_nan=True, time_index=time_idx)

        return pd.DataFrame(
            np.hstack((data_coverage.reshape((-1, 1)), data_coverage_n.reshape((-1, 1)))),
            columns=["coverage", "coverage_n"],
            index=time_agg,
        )

    def determine_coverage(
        self, data_observations: np.ndarray, time_index: Union[None, dict], mask_raster: Union[None, np.ndarray]
    ) -> Tuple[np.ndarray, list]:
        """
        Calculate the spatial coverage for each time slice. The spatial coverage is defined as the number of pixels with
        with a numeric value in the data-cube divided by the number of pixels in the mask over the same number of time
        points. If no mask is provided the coverage is calculated against the total pixels.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        mask_raster : np.ndarray
            The binary mask used on the dataset.
        time_index : Union[None, dict]
            The time indexes for each time slice to be processed.

        Returns
        -------
        Tuple[np.ndarray, list]
            The spacial coverage for each time slice and the time index.
        """
        if time_index is None:
            time_index = {0: [0]}

        data_coverage = []
        time_coords_agg = []
        for time_point in time_index.keys():
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            data_coverage.append(
                self.determine_coverage_point(
                    data_observations=data_observations[time_min:time_max, :, :], mask_raster=mask_raster
                )
            )
        return np.array(data_coverage), time_coords_agg

    def determine_coverage_point(
        self: Any, data_observations: np.ndarray, mask_raster: Union[None, np.ndarray]
    ) -> float:
        """
        Calculate the coverage for a specific data cube. If no mask is provided the coverage is calculated against the
        total pixels.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        mask_raster : np.ndarray
            The rasterised mask.

        Returns
        -------
        float
            The spatial coverage.
        """
        n_observations = self.count_pixels(data=data_observations)
        # Multiply by the number of time points.
        n_raster = self.count_pixels(data=mask_raster) * data_observations.shape[0]

        return n_observations / n_raster

    @staticmethod
    def count_pixels(data: np.ndarray) -> Union[float, int]:
        """
        Count the number of valid pixels. In the case of a boolean mask this is the number of False elements. In all
        other cases this is the number of non-NaN elements. If no mask is provided the coverage is calculated against
        the total pixels.

        Parameters
        ----------
        data : np.ndarray
            The data cube.

        Returns
        -------
        Union[float, int]
            The number of elements which are False or non-NaN.
        """
        # Assumes that False in mask are included pixels.
        if data.dtype == bool:
            return np.count_nonzero(~data)
        else:
            return np.count_nonzero(~np.isnan(data))
