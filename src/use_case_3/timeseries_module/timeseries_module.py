"""Module for processing data into timeseries."""

from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skgstat as skg
import xarray as xr
from scipy.stats import bootstrap
from sklearn.linear_model import LinearRegression

# from use_case_3.helper_functions import *


class TimeSeriesModule:
    """The time series module for converting a datacube to a 1D time series and calculating summary statistics."""

    def __init__(self) -> None:
        # These are all uses to check whether parameters passed are allowed.
        self.aggregation_types = ("mean", "median", "max", "min")
        self.uncertainty_types = ("std", "max", "min", "IQR")
        self.periods = ("month", "year")
        self.normalisation_types = ("first", "linear_regression")
        self.plot_spread = (None, "uncertainties", "percentiles")


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
            The reordered data.
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
            The reordered data and dimension names.
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
            Returns the dictionary, mapping each time slice to the data-cube indexes.

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
        dimension_data : Union[xr.DataArray, pd.DataFrame, dict]
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

    def mask_data_cube(
        self: Any,
        data_cube: np.ndarray,
        data_mask: np.ndarray,
        dimension_names: List[str],
        modify: bool,
    ) -> Tuple[np.ndarray, bool]:
        """
        Mask a data cube based on a given mask. This makes some strong assumptions on the data and mask:
            - 1) The data mask and the data are both in the same projection.
            - 2) The data mask and the data are both either y_affine_negative or y_affine_positive.
            - 3) The data cube will be rearranged so that the time axis is the first axis, however no checks will be
                 done that the second and third axis will be the same as the mask. There is a check on the dimensions,
                 but in the case of the mask being square, the dimensions could be wrong.

        Parameters
        ----------
        data_cube : np.ndarray
            The data cube to be masked.
        data_mask : np.ndarray
            The rasterised data mask.
        dimension_names : List[str]
            A list of the dimension names. These need to be in the order that they index the axes of the datacube.
        modify : bool
            Whether to modify the dimensions of the datacube.

        Returns
        -------
        np.ndarray
            The masked data cube.

        Raises
        ------
        ValueError
            The dimensions of the mask and the second and third axis of the datacube don't match.
        """
        data_cube, dimension_names = self._expand_dimensions(
            data=data_cube, dimension_names=dimension_names, modify_dimension_names=modify
        )

        data_cube = self._move_axes_time(data=data_cube, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        nan_array = np.empty((data_cube.shape[1], data_cube.shape[2]))
        nan_array[:] = np.nan

        if data_mask.shape != (data_cube.shape[1], data_cube.shape[2]):
            raise ValueError(
                f"Your data {data_cube.shape[1], data_cube.shape[2]} and mask {data_mask.shape} shapes are different"
            )

        data_cube_masked = np.where(data_mask, nan_array, data_cube)

        if modify:
            modify = False

        return data_cube_masked, modify

    def process_timeseries(
        self: Any,
        data_observations: np.ndarray,
        data_uncertainties: Union[None, np.ndarray],
        data_mask: Union[None, np.ndarray],
        apply_mask: bool,
        dimension_data: dict,
        dimension_names: List[str],
        aggregate: str = "mean",
        percentiles: Tuple[float, float] = (25, 75),
        period: Union[None, str] = None,
        uncertainty: Union[None, str] = None,
        independent: bool = False,
        t0_normalisation: Union[None, str] = None,
    ) -> pd.DataFrame:
        """
        Process the input data into a time series with associated uncertainties and coverage. This is the main method
        processing the datacube into the time series.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        data_uncertainties : Union[None, np.ndarray]
            The data uncertainties.
        data_mask : Union[None, np.ndarray]
            The data mask.
        apply_mask : bool
            Boolean flag whereby if True the mask supplied to the data will be used to mask the data. If False then the
            mask is only used to calculate the coverage.
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
        percentiles : Tuple[float, float]
            The upper and lower percentiles to calculate. These are of the form (percentile_lower, and percentile_upper)
        period : Union[None, str], optional
            The time period over which to aggregate, by default None
        uncertainty : Union[None, str], optional
            The statistic type describing the uncertainty, by default None
        independent : bool, optional
            Whether the uncertainties are independent or not, by default False
        t0_normalisation : Union[None, str], optional
            Whether to normalise the time series against time zero, by default None

        Returns
        -------
        pd.DataFrame
            The aggregated 1D time series as a DataFrame. The DataFrame contains as columns the aggregated observation,
            the coverage calculated against the supplied mask, the upper and lower percentiles, and optionally if
            specified then the uncertainties.
        """
        if apply_mask:
            data_observations, modify = self.mask_data_cube(
                data_cube=data_observations, data_mask=data_mask, dimension_names=dimension_names, modify=True
            )
            if data_uncertainties is not None:
                data_uncertainties, modify = self.mask_data_cube(
                    data_cube=data_uncertainties, data_mask=data_mask, dimension_names=dimension_names, modify=modify
                )
        else:
            modify = True

        data_agg = self.calculate_aggregate_timeseries(
            data_observations=data_observations,
            dimension_data=dimension_data,
            dimension_names=dimension_names,
            aggregate=aggregate,
            period=period,
            t0_normalisation=t0_normalisation,
            modify=modify,
        )

        data_coverage_agg = self.calculate_coverage_timeseries(
            data_observations=data_observations,
            data_mask=data_mask,
            dimension_data=dimension_data,
            dimension_names=dimension_names,
            period=period,
        )

        data_percentiles_agg = self.calculate_percentiles_timeseries(
            data_observations=data_observations,
            dimension_data=dimension_data,
            dimension_names=dimension_names,
            period=period,
            percentiles=percentiles,
        )

        data_timeseries = pd.merge(data_agg, data_coverage_agg, left_index=True, right_index=True)
        data_timeseries = pd.merge(data_timeseries, data_percentiles_agg, left_index=True, right_index=True)
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
                modify=modify,
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
        modify: bool = False,
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
        period : Union[None, str], optional
            The time period over which to aggregate, by default None.
        t0_normalisation : Union[None, str], optional
            Whether to normalise the time series against time zero, by default None.
        modify : bool
            Whether to modify the dimensions of the datacube, by default False.

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
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=modify
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
            columns=["aggregation", "data_" + aggregate_spread, "aggregation_n"],
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
        modify: bool = False,
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
        period : Union[None, str], optional
            The time period over which to aggregate, by default None.
        independent : bool, optional
            Whether the uncertainties are independent or not, by default False
        t0_normalisation : Union[None, str], optional
            Whether to normalise the time series against time zero, by default None.
        modify : bool
            Whether to modify the dimensions of the datacube, by default False.

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
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=modify
        )
        data_uncertainties, dimension_names = self._expand_dimensions(
            data=data_uncertainties, dimension_names=dimension_names, modify_dimension_names=modify
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
        time_index : Union[None, dict], optional
            The time indexes over which to do the aggregations, by default None

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
        aggregate : str, optional
            The type of aggregation performed on the data, by default "mean". Currently these are limited to:
            mean, median, max, min. Note, the aggregation chosen determines the spread according to the following rules:
                - mean: standard deviation
                - median: IQR
                - max/min: NaN
        time_index : Union[None, dict], optional
            The time indexes over which to do the aggregations, by default None
        dimension_names : list
            Specification of the dimension names.
        dimension_aggregate : Union[None, int, Tuple[int]], optional
            The dimension of the data array to aggregate against, by default None.

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
        time_index : Union[None, dict], optional
            The time indexes over which to do the aggregations, by default None
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
                    data_propagated_space = np.zeros((time_max - time_min))
                    for time_idx in range(time_min, time_max):
                        data_agg_n_time_point = np.count_nonzero(~np.isnan(data_uncertainties[time_idx, :, :]))
                        data_propagated_space[time_idx - time_min] = (
                            np.nansum(data_uncertainties[time_min:time_max, :, :]) / data_agg_n_time_point
                        )
                    data_agg.append(np.sqrt(np.nansum(np.square(data_propagated_space))) / (time_max - time_min))

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
        time_index : Union[None, dict], optional
            The time indexes over which to do the aggregations, by default None

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
        data_uncertainties_1: Union[np.float64, np.ndarray], data_uncertainties_2: Union[np.float64, np.ndarray]
    ) -> Union[np.float64, np.ndarray]:
        """
        Combine uncertainties through quadrature.

        Parameters
        ----------
        data_uncertainties_1 : Union[np.float64, np.ndarray]
            The first vector of uncertainties.
        data_uncertainties_2 : Union[np.float64, np.ndarray]
            The second vector of uncertainties.

        Returns
        -------
        Union[np.float64, np.ndarray]
            The combined uncertainty vector.
        """
        return np.sqrt(np.square(data_uncertainties_1) + np.square(data_uncertainties_2))

    def calculate_percentiles_timeseries(
        self: Any,
        data_observations: np.ndarray,
        dimension_data: dict,
        dimension_names: List[str],
        period: Union[None, str] = None,
        percentiles: Tuple[float, float] = (25, 75),
    ) -> pd.DataFrame:
        """
        Calculate the upper and lower percentiles of a datacube of observations.

        Parameters
        ----------
        data_observations : np.ndarray
            The datacube of observations.
        dimension_data : dict
            The coordinate data.
        dimension_names : List[str]
            The dimension names
        period : Union[None, str], optional
            The period over which the data is aggregated, by default None
        percentiles : Tuple[float, float], optional
            The lower and upper percentile to be calculated for a datacube. These must be a tuple of floats in the range
            [0.0, 100.0], by default (25, 75)

        Returns
        -------
        pd.DataFrame
            A dataframe of the upper/lower percentiles and the number of data points considered in each time slice. The
            index of the dataframe is the first time point for a given time slice.
        """
        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=True
        )

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        time_idx = self.process_time_indexes(
            dimension_names=dimension_names, dimension_data=dimension_data, period=period
        )

        data_percentile_lower, time_agg = self.determine_percentiles(
            data_observations=data_observations, time_index=time_idx, percentile=percentiles[0]
        )

        data_percentile_upper, _ = self.determine_percentiles(
            data_observations=data_observations, time_index=time_idx, percentile=percentiles[1]
        )

        data_percentile_n = self.count_nan_elements(data=data_observations, non_nan=True, time_index=time_idx)

        return pd.DataFrame(
            np.hstack(
                (
                    data_percentile_lower.reshape((-1, 1)),
                    data_percentile_upper.reshape((-1, 1)),
                    data_percentile_n.reshape((-1, 1)),
                )
            ),
            columns=[f"percentile_{percentiles[0]}", f"percentile_{percentiles[1]}", "percentile_n"],
            index=time_agg,
        )

    def determine_percentiles(
        self: Any, data_observations: np.ndarray, time_index: Union[None, dict], percentile: float
    ) -> Tuple[np.ndarray, list]:
        """
        Calculate the percentile for a given datacube and time index.

        Parameters
        ----------
        data_observations : np.ndarray
            The data cube of observations.
        time_index : Union[None, dict]
            The time indexes for each time slice to be processed.
        percentile : float
            The percentile to be calculated. This needs to be a float in the range [0.0, 100.0] inclusive.

        Returns
        -------
        Tuple[np.ndarray, list]
            The percentiles and associated time coordinates for each percentile.
        """
        if time_index is None:
            time_index = {0: [0]}

        data_percentile = []
        time_coords_agg = []
        for time_point in time_index.keys():
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            data_percentile.append(np.nanpercentile(data_observations[time_min:time_max, :, :], percentile))
        return np.array(data_percentile), time_coords_agg

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
        data_observations : np.ndarray
            The data observations.
        dimension_data : dict
            The coordinate data.
        dimension_names : List[str]
            The dimension names
        data_mask : Union[None, np.ndarray], optional
            The data mask, by default None
        period : Union[None, str], optional
            The period over which the data is aggregated, by default None

        Returns
        -------
        pd.DataFrame
            The time series of the coverage.
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
            np.hstack((data_coverage.reshape((-1, 1)) * 100, data_coverage_n.reshape((-1, 1)))),
            columns=["coverage", "coverage_n"],
            index=time_agg,
        )

    def determine_coverage(
        self, data_observations: np.ndarray, time_index: Union[None, dict], mask_raster: np.ndarray
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

    def determine_coverage_point(self: Any, data_observations: np.ndarray, mask_raster: np.ndarray) -> float:
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

    def summary_statistics(
        self: Any, timeseries: pd.DataFrame, observation_name: str, observation_units: str
    ) -> pd.DataFrame:
        """
        Generate a set of summary statistics for a time series. This assumes that the timeseries is
        ordered. It also assumes that there are associated uncertainties.

        Parameters
        ----------
        timeseries : pd.DataFrame
            The time series.
        observation_name : str
            The name of the observation.
        observation_units : str
            The units of the observation.

        Returns
        -------
        pd.DataFrame
            A set of summary statistics.
        """
        start = timeseries["aggregation"].iloc[0]
        start_uncertainty = timeseries["uncertainty"].iloc[0]
        end = timeseries["aggregation"].iloc[-1]
        end_uncertainty = timeseries["uncertainty"].iloc[-1]
        difference = end - start
        difference_uncertainty = self.combine_uncertainties_quadrature(start_uncertainty, end_uncertainty)

        value_maximum = timeseries["aggregation"].max()
        value_maximum_uncertainty = timeseries.loc[timeseries["aggregation"].idxmax(), "uncertainty"]

        value_minimum = timeseries["aggregation"].min()
        value_minimum_uncertainty = timeseries.loc[timeseries["aggregation"].idxmin(), "uncertainty"]
        percentile_90 = np.percentile(timeseries["aggregation"], 90)
        percentile_75 = np.percentile(timeseries["aggregation"], 75)
        percentile_50 = np.percentile(timeseries["aggregation"], 50)
        percentile_25 = np.percentile(timeseries["aggregation"], 25)
        percentile_10 = np.percentile(timeseries["aggregation"], 10)
        coverage_maximum = timeseries["coverage"].max()
        coverage_minimum = timeseries["coverage"].min()
        coverage_range = coverage_maximum - coverage_minimum
        coverage_mean = timeseries["coverage"].mean()
        coverage_std = timeseries["coverage"].std()
        coverage_median = timeseries["coverage"].median()

        observation_uncertainty_corr = np.corrcoef(
            timeseries["aggregation"].values.flatten(), timeseries["uncertainty"].values.flatten()
        )[0, -1]
        linearity_corr = np.corrcoef(np.arange(timeseries.shape[0]), timeseries["aggregation"].values.flatten())[0, -1]

        data = np.zeros((18, 3))

        # Start
        data[0, 0] = start
        data[0, 1] = start - start_uncertainty
        data[0, 2] = start + start_uncertainty

        # End
        data[1, 0] = end
        data[1, 1] = end - end_uncertainty
        data[1, 2] = end + end_uncertainty

        # Difference
        data[2, 0] = difference
        data[2, 1] = difference - difference_uncertainty
        data[2, 2] = difference + difference_uncertainty

        # Maximum
        data[3, 0] = value_maximum
        data[3, 1] = value_maximum - value_maximum_uncertainty
        data[3, 2] = value_maximum + value_maximum_uncertainty

        # Minimum
        data[4, 0] = value_minimum
        data[4, 1] = value_minimum - value_minimum_uncertainty
        data[4, 2] = value_minimum + value_minimum_uncertainty

        # Percentiles (0.9, 0.75, 0.5, 0.25, 0.1)
        data[5, 0] = percentile_10
        data[5, 1] = np.nan
        data[5, 2] = np.nan

        data[6, 0] = percentile_25
        data[6, 1] = np.nan
        data[6, 2] = np.nan

        data[7, 0] = percentile_50
        data[7, 1] = np.nan
        data[7, 2] = np.nan

        data[8, 0] = percentile_75
        data[8, 1] = np.nan
        data[8, 2] = np.nan

        data[9, 0] = percentile_90
        data[9, 1] = np.nan
        data[9, 2] = np.nan

        # Coverage Maximum
        data[10, 0] = coverage_maximum
        data[10, 1] = np.nan
        data[10, 2] = np.nan

        # Coverage Minimum
        data[11, 0] = coverage_minimum
        data[11, 1] = np.nan
        data[11, 2] = np.nan

        # Coverage Range
        data[12, 0] = coverage_range
        data[12, 1] = np.nan
        data[12, 2] = np.nan

        # Coverage Mean
        data[13, 0] = coverage_mean
        data[13, 1] = np.nan
        data[13, 2] = np.nan

        # Coverage Median
        data[14, 0] = coverage_median
        data[14, 1] = np.nan
        data[14, 2] = np.nan

        # Coverage Std
        data[15, 0] = coverage_std
        data[15, 1] = np.nan
        data[15, 2] = np.nan

        # observation_uncertainty_corr
        data[16, 0] = observation_uncertainty_corr
        data[16, 1] = np.nan
        data[16, 2] = np.nan

        # linearity_corr
        data[17, 0] = linearity_corr
        data[17, 1] = np.nan
        data[17, 2] = np.nan

        indexes = [
            f"{observation_name} - Initial Value ({observation_units})",
            f"{observation_name} - Final Value ({observation_units})",
            f"{observation_name} - Difference ({observation_units})",
            f"{observation_name} - Maximum Value ({observation_units})",
            f"{observation_name} - Minimum Value ({observation_units})",
            f"{observation_name} - 10th Percentile Value ({observation_units})",
            f"{observation_name} - 25th Percentile Value ({observation_units})",
            f"{observation_name} - 50th Percentile Value ({observation_units})",
            f"{observation_name} - 75th Percentile Value ({observation_units})",
            f"{observation_name} - 90th Percentile Value ({observation_units})",
            "Coverage Maximum (%)",
            "Coverage Minimum (%)",
            "Coverage Range (%)",
            "Coverage Mean (%)",
            "Coverage Median (%)",
            "Coverage Standard Deviation (%)",
            "Observation/Uncertainty Correlation",
            "Observation Linearity (correlation coef.)",
        ]

        summary_statistics = pd.DataFrame(
            data=data, index=indexes, columns=["Value", "Lower Uncertainty Bound", "Upper Uncertainty Bound"]
        )

        return summary_statistics

    def process_spatial_correlation(
        self: Any,
        data_observations: np.ndarray,
        dimension_data: dict,
        dimension_names: List[str],
    ) -> np.ndarray:  # noqa
        """
        Calculate the spatial correlation for each time point.

        Parameters
        ----------
        data_observations : np.ndarray
            The datacube.
        dimension_data : dict
            The dimension data.
        dimension_names : List[str]
            The dimension names indexing the datacube and the dimension_data.

        Returns
        -------
        np.ndarray
            The spatial dependency index for each timepoint.
        """
        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=True
        )

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        data_slices = self.flatten_data_cube(
            data_cube=data_observations, dimension_data=dimension_data, dimension_names=dimension_names
        )

        sdi = np.zeros((len(data_slices), 1))
        for i, data_slice in enumerate(data_slices):
            variogram = skg.Variogram(data_slice[:, :2], data_slice[:, -1], use_nugget=True, n_lags=50, maxlag="median")
            sdi[i] = (variogram.parameters[1] / (variogram.parameters[-1] + variogram.parameters[1])) * 100

        return sdi

    def flatten_data_cube(
        self: Any, data_cube: np.ndarray, dimension_data: dict, dimension_names: List[str], exclude_nan: bool = True
    ) -> List[np.ndarray]:
        """
        Flatten a data_cube from a 3d array to a 2d array of the form (time, [x, y, value]).

        Parameters
        ----------
        data_cube : np.ndarray
            The input datacube of the form time, projection_y_coordinate, projection_x_coordinate.
        dimension_data : dict
            The dimension data
        dimension_names : List[str]
            The dimension names.
        exclude_nan : bool, optional
            Whether any nan values should be ignored, by default True

        Returns
        -------
        List[np.ndarray]
            The flattened datacube. This is a list containing the flattened data for each time point.
        """
        # flatten the data spatially but not in time, assuming the time dimension is first.
        data_slices = []

        for i in range(data_cube.shape[0]):
            data_slice = data_cube[i, :, :].flatten().reshape((-1, 1))
            coord_1 = (
                np.resize(
                    dimension_data[dimension_names[1]],
                    new_shape=(
                        dimension_data[dimension_names[1]].shape[0],
                        dimension_data[dimension_names[2]].shape[0],
                    ),
                )
                .flatten(order="F")
                .reshape((-1, 1))
            )

            coord_2 = (
                np.resize(
                    dimension_data[dimension_names[2]],
                    new_shape=(
                        dimension_data[dimension_names[2]].shape[0],
                        dimension_data[dimension_names[1]].shape[0],
                    ),
                )
                .flatten(order="C")
                .reshape((-1, 1))
            )

            data_slice = np.hstack((coord_1, coord_2, data_slice))
            if exclude_nan:
                data_slice = data_slice[~np.isnan(data_slice).any(axis=1)]
            if data_slice.shape[0] > 0:
                data_slices.append(data_slice)

        return data_slices

    def plot(
        self: Any,
        data_timeseries: pd.DataFrame,
        spread: Union[None, str],
        percentiles: Union[None, Tuple[int, int]] = None,
        x_label: Union[None, str] = None,
        y_label: Union[None, str] = None,
    ) -> None:
        """
        Plot the timeseries data.

        Parameters
        ----------
        data_timeseries : pd.DataFrame
            The timeseries.
        spread : Union[None, str]
            The spread to display on the line plot. This can be one of None, "uncertainties" and "percentiles".
        percentiles : Union[None, Tuple[int, int]], optional
            The percentiles, by default None
        x_label : Union[None, str], optional
            A X-axis label to use, by default None
        y_label : Union[None, str], optional
            A Y-axis label to use, by default None

        Raises
        ------
        ValueError
            f"the spread needs to be one of {self.plot_spread}"
        ValueError
            "Percentiles need to be specified!"
        ValueError
            "Percentiles do not match the columns in the timeseries!"
        """
        error_pos = None
        error_neg = None

        if spread not in self.plot_spread:
            raise ValueError(f"the spread needs to be one of {self.plot_spread}")

        if spread == "uncertainties":
            error_pos = data_timeseries["aggregation"] + data_timeseries["uncertainty"]
            error_neg = data_timeseries["aggregation"] - data_timeseries["uncertainty"]
        elif spread == "percentiles":
            if percentiles is None:
                raise ValueError("Percentiles need to be specified!")

            if (
                f"percentile_{percentiles[0]}" not in data_timeseries.columns
                or f"percentile_{percentiles[1]}" not in data_timeseries.columns
            ):
                raise ValueError("Percentiles do not match the columns in the timeseries!")
            error_pos = data_timeseries[f"percentile_{percentiles[0]}"]
            error_neg = data_timeseries[f"percentile_{percentiles[1]}"]

        fig, axes = plt.subplots(1, 1)
        # sns.set(font_scale=2)
        sns.lineplot(
            x=data_timeseries.index,
            y=data_timeseries["aggregation"],
            color="black",
            ax=axes,
            linewidth=0.5,
        )
        sns.scatterplot(
            x=data_timeseries.index,
            y=data_timeseries["aggregation"],
            color="black",
            ax=axes,
            s=10,
            linewidth=0.5,
        )

        if spread is not None:
            sns.lineplot(
                x=data_timeseries.index,
                y=error_pos,
                color="black",
                alpha=0.5,
                linewidth=0.5,
                linestyle="--",
                ax=axes,
            )

            sns.lineplot(
                x=data_timeseries.index,
                y=error_neg,
                color="black",
                alpha=0.5,
                linewidth=0.5,
                linestyle="--",
                ax=axes,
            )
            plt.fill_between(
                data_timeseries.index,
                error_pos,
                error_neg,
                color="black",
                alpha=0.25,
            )
        plt.grid()
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        plt.show()
