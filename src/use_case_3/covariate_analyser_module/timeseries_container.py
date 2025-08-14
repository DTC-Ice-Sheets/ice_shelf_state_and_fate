"""Module for the manipulation of timeseries."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class TimeSeriesContainer:
    """Module for the manipulation of timeseries."""

    def __init__(
        self: TimeSeriesContainer,
        # timeseries: dict[str: Timeseries] = {}
        timeseries: list[pd.Series],
        timeseries_labels: list[str],
        timeseries_units: list[str],
    ) -> None:
        """
        Initialise and create the necessary timeseries container.

        Parameters
        ----------
        timeseries : list[pd.Series]
            The timeseries as pd.Series.
        timeseries_labels : list[str]
            The timeseries labels. These must be unique.
        timeseries_units : list[str]
            The units of the timeseries.
        """
        self.timeseries = {}
        self.add_timeseries(
            timeseries=timeseries, timeseries_labels=timeseries_labels, timeseries_units=timeseries_units
        )

    def prepare_timeseries(
        self: TimeSeriesContainer,
        time_period: str,
        timeseries_labels: list[str] | None = None,
        aggregation: str = "mean",
        inplace: bool = True,
    ) -> list[pd.Series] | None:
        """
        Prepare the timeseries for the covariate analysis. This first checks that all the timeseries intersect in time,
        it then resamples the timeseries to be the same frequency, trims the time series to be the same length and then
        interpolates any missing values using linear interpolation.

        Parameters
        ----------
        time_period : str
            The time period.
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        aggregation : str, optional
            The aggregation to be applied, by default "mean"
        inplace : bool, optional
            Whether the timeseries are modified in place, or instead returned as a list of timeseries.

        Returns
        -------
        list[pd.Series] | None
            Either a list of the processed timeseries or None depending on whether inplace is set.
        """
        if time_period == "M":
            period = "MS"
        elif time_period == "Q":
            period = "QS"
        elif time_period == "Y":
            period = "YS"
        elif time_period == "d":
            period = "d"

        if timeseries_labels is None:
            timeseries, timeseries_labels = self.get_timeseries_all()
        else:
            timeseries = self.get_timeseries(timeseries_labels=timeseries_labels)

        if not self._check_timeseries_intersection(timeseries=timeseries):
            raise ValueError("One or more of the specified time series don't intersect temporaly")

        timeseries = self._resample_timeseries(timeseries=timeseries, time_period=time_period, aggregation=aggregation)

        start_date, end_date = self._find_start_end_dates(timeseries=timeseries)

        timeseries = self._trim_timeseries_date(timeseries=timeseries, start_date=start_date, end_date=end_date)

        timeseries = self._interpolate_timeseries(timeseries=timeseries, period=period)

        if inplace:
            self.add_timeseries(
                timeseries=timeseries,
                timeseries_labels=timeseries_labels,
                timeseries_units=self.get_units_all()[0],
                overwrite=True,
            )
        else:
            return timeseries

    def _combine_timeseries_from_labels_and_list(
        self: TimeSeriesContainer, timeseries_labels: list[str] | None = None, timeseries: list[pd.Series] | None = None
    ) -> list[pd.Series]:
        """
        Combine lists of timeseries collected using either labels or directly as a passed list. If no list of labels or
        list of timeseries is passed, the default behavior is to return all of the timeseries in this container.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The list of timeseries, by default None
        timeseries : list[pd.Series] | None, optional
            The list of timeseries labels, by default None

        Returns
        -------
        list[pd.Series]
            The total list of timeseries.
        """
        if timeseries_labels is not None:
            timeseries_from_labels = self.get_timeseries(timeseries_labels=timeseries_labels)
        elif timeseries_labels is None and timeseries is None:
            timeseries_from_labels, timeseries_labels = self.get_timeseries_all()
        else:
            timeseries_from_labels = []

        if timeseries is not None:
            timeseries += timeseries_from_labels
        else:
            timeseries = timeseries_from_labels

        return timeseries

    def _check_timeseries_intersection(
        self: TimeSeriesContainer, timeseries_labels: list[str] | None = None, timeseries: list[pd.Series] | None = None
    ) -> bool:
        """
        Determine whether all time series intersect in time by at least one time point. If both input parameters are
        None then it will check all of the timeseries in the container.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[pd.Series] | None, optional
            The list of time series, by default None

        Returns
        -------
        bool
            True if all time series intersect with at least one time point.
        """
        timeseries = self._combine_timeseries_from_labels_and_list(
            timeseries_labels=timeseries_labels, timeseries=timeseries
        )

        start_datetime, end_datetime = self._find_start_end_dates(timeseries=timeseries)

        return end_datetime >= start_datetime

    def _find_start_end_dates(
        self: TimeSeriesContainer, timeseries_labels: list[str] | None = None, timeseries: list[pd.Series] | None = None
    ) -> tuple[pd.Timestamp | np.datetime64, pd.Timestamp | np.datetime64]:
        """
        Return the greatest common start and the smallest common end time for a set of time series. If both input
        parameters are None then it will check all of the timeseries in the container.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[pd.Series] | None, optional
            The list of time series, by default None

        Returns
        -------
        tuple[pd.Timestamp | np.datetime64, pd.Timestamp | np.datetime64]
            The start and end datetime. Note it is possible for the end time to be less than the start time in cases
            where the time series do not intersect in time
        """
        timeseries = self._combine_timeseries_from_labels_and_list(
            timeseries_labels=timeseries_labels, timeseries=timeseries
        )

        # Find the latest start datetime for all the input timeseries
        start_date = np.max([np.min(series.index) for series in timeseries])
        # Find the earliest end datetime for all the input timeseries
        end_date = np.min([np.max(series.index) for series in timeseries])

        return start_date, end_date

    def _trim_timeseries_date(
        self: TimeSeriesContainer,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        timeseries_labels: list[str] | None = None,
        timeseries: list[pd.Series] | None = None,
    ) -> list[pd.Series]:
        """
        Trim a time series using a start and end date time. If both input parameters are None then it will check all of
        the timeseries in the container.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[pd.Series] | None, optional
            The list of time series, by default None
        start_date : int
            The start date to trim.
        end_date : int
            The end date to trim.

        Returns
        -------
        list[pd.Series] | None
            The trimmed timeseries with minimum time start_date and maximum time end_date
        """
        timeseries = self._combine_timeseries_from_labels_and_list(
            timeseries_labels=timeseries_labels, timeseries=timeseries
        )

        timeseries = [df[(df.index >= start_date) & (df.index <= end_date)] for df in timeseries]

        return timeseries

    def _resample_timeseries(
        self: TimeSeriesContainer,
        time_period: str,
        timeseries_labels: list[str] | None = None,
        timeseries: list[pd.Series] | None = None,
        aggregation: str = "mean",
    ) -> list[pd.Series]:
        """
        Resample the 1D timeseries into specific time periods (days, weeks, months, quarters, years).

        Parameters
        ----------
        time_period : str
            The time period.
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[pd.Series] | None, optional
            The list of time series, by default None
        aggregation : str, optional
            The aggregation to be applied, by default "mean"

        Returns
        -------
        list[pd.Series]
            The aggregated time series.
        """
        timeseries = self._combine_timeseries_from_labels_and_list(
            timeseries_labels=timeseries_labels, timeseries=timeseries
        )

        timeseries_agg_list = []
        for timeseries_single in timeseries:
            if aggregation == "mean":
                timeseries_agg = timeseries_single.groupby(
                    pd.PeriodIndex(timeseries_single.index, freq=time_period)
                ).mean()
            elif aggregation == "median":
                timeseries_agg = timeseries_single.groupby(
                    pd.PeriodIndex(timeseries_single.index, freq=time_period)
                ).median()
            elif aggregation == "max":
                timeseries_agg = timeseries_single.groupby(
                    pd.PeriodIndex(timeseries_single.index, freq=time_period)
                ).max()
            elif aggregation == "min":
                timeseries_agg = timeseries_single.groupby(
                    pd.PeriodIndex(timeseries_single.index, freq=time_period)
                ).min()
            timeseries_agg.index = timeseries_agg.index.to_timestamp()
            timeseries_agg_list.append(timeseries_agg)

        return timeseries_agg_list

    def _interpolate_timeseries(
        self: TimeSeriesContainer,
        period: str,
        timeseries_labels: list[str] | None = None,
        timeseries: list[pd.Series] | None = None,
    ) -> list[pd.Series]:
        """
        Resamples any missing data and then linearly interpolates a timeseries according to a specified period.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[pd.Series] | None, optional
            The list of time series, by default None
        period : str
            The period to resample the time series.

        Returns
        -------
        pd.Series
            The resampled and interpolated time series.
        """
        timeseries = self._combine_timeseries_from_labels_and_list(
            timeseries_labels=timeseries_labels, timeseries=timeseries
        )

        timeseries_interpolated = []
        for timeseries_single in timeseries:
            # Generate the list of date idx
            date_idx = pd.date_range(
                start=timeseries_single.index[0],
                end=timeseries_single.index[-1],
                freq=period,
            )

            timeseries_interpolated.append(
                timeseries_single.reindex(date_idx, fill_value=np.nan).interpolate(method="linear")
            )

        return timeseries_interpolated

    def _infer_frequency(
        self: TimeSeriesContainer,
        timeseries_label: str | None = None,
        timeseries: pd.Series | None = None,
    ) -> str:
        """
        Infer the frequency between period index and return this as a string. Note only one of either the timeseries or
        label can be provided.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[pd.Series] | None, optional
            The list of time series, by default None

        Returns
        -------
        str
            The period as a string.
        """
        if timeseries is None and timeseries_label is None:
            raise ValueError("Both the timeseries and the timeseries_label parameters cannot be None")

        if timeseries is not None and timeseries_label is not None:
            raise ValueError(
                "Both the timeseries and the timeseries_label parameters cannot be used at once, please choose one"
            )

        if timeseries_label is not None:
            timeseries = self.get_timeseries(timeseries_labels=[timeseries_label])[0]

        timeseries = timeseries.copy()
        if not isinstance(timeseries.index, pd.PeriodIndex):
            timeseries.index = timeseries.index.to_period()
        freqstr = timeseries.index.freq.freqstr
        if freqstr[0] == "M":
            freq = "Month"
        elif freqstr[0] == "Q":
            freq = "Quarter"
        elif freqstr[0] == "Y":
            freq = "Year"

        return freq

    def add_timeseries(
        self: TimeSeriesContainer,
        timeseries: list[pd.Series],
        timeseries_labels: list[str],
        timeseries_units: list[str],
        overwrite: bool = False,
    ) -> None:
        """
        Set all the time series. It is required that labels and units are also provided for every timeseries.

        Parameters
        ----------
        timeseries : list[pd.Series]
            The timeseries as a dictionary where the key is a label in the list of timeseries labels.
        timeseries_labels : list[str]
            The timeseries labels. These are the keys for the timeseries and associated units.
        timeseries_units : list[str]
            The timeseries units. If a time series is unit-less then use an empty string.
        overwrite : bool, optional
            if set to True all the timeseries, labels and units will be overwritten. If False then these
            will only be added if no time series have been added already, by default False

        Raises
        ------
        ValueError
            Timeseries data already present and overwrite is set to False
        """
        for i, timeseries_label in enumerate(timeseries_labels):
            if timeseries_label in self.timeseries.keys():
                label_found = True
            else:
                label_found = False

            if not overwrite and label_found:
                raise ValueError("Timeseries is already added")
            else:
                self.timeseries[timeseries_label] = [timeseries[i], timeseries_units[i]]

    @staticmethod
    def _sanity_check_inputs(
        timeseries: list[pd.Series], timeseries_labels: list[str], timeseries_units: list[str]
    ) -> bool:
        """
        Check that all of the input lists have the same number of elements.

        Parameters
        ----------
        timeseries : list[pd.Series]
            The timeseries as a dictionary where the key is a label in the list of timeseries labels.
        timeseries_labels : list[str]
            The timeseries labels. These are the keys for the timeseries and associated units.
        timeseries_units : list[str]
            The timeseries units. If a time series is unit-less then use an empty string.

        Returns
        -------
        bool
            True if all inputs have the same number of elements.
        """
        if len(timeseries) != len(timeseries_labels) or len(timeseries_units) != len(timeseries_labels):
            return False
        else:
            return True

    def remove_timeseries(self, timeseries_labels: list[str]) -> None:
        """
        Remove the specified timeseries.

        Parameters
        ----------
        label : list[str]
            The time series label.

        Raises
        ------
        ValueError
            "There is no timeseries with this label"
        """
        for timeseries_label in timeseries_labels:
            if timeseries_label in self.timeseries.keys():
                del self.timeseries[timeseries_label]
            else:
                raise ValueError("There is no timeseries with this label")

    def get_timeseries(self: TimeSeriesContainer, timeseries_labels: list[str]) -> list[pd.Series]:
        """
        Get the specified timeseries.

        Parameters
        ----------
        label : list[str]
            The timeseries name.

        Returns
        -------
        list[pd.Series]
            The timeseries.

        Raises
        ------
        ValueError
            "There is no timeseries with this name"
        """
        timeseries = []
        for timeseries_label in timeseries_labels:
            if timeseries_label in self.timeseries.keys():
                timeseries.append(self.timeseries[timeseries_label][0])
            else:
                raise ValueError("There is no timeseries with this name")

        return timeseries

    def get_units(self: TimeSeriesContainer, timeseries_labels: list[str]) -> list[str]:
        """
        Get the unit associated with the timeseries.

        Parameters
        ----------
        label : list[str]
            The timeseries label.

        Returns
        -------
        str
            The timeseries unit.

        Raises
        ------
        ValueError
            "There is no timeseries with this name"
        """
        timeseries_units = []
        for timeseries_label in timeseries_labels:
            if timeseries_label in self.timeseries.keys():
                timeseries_units.append(self.timeseries[timeseries_label][1])
            else:
                raise ValueError("There is no timeseries with this name")

        return timeseries_units

    def get_timeseries_all(self: TimeSeriesContainer) -> tuple[list[pd.Series], list[str]]:
        """
        Get all the timeseries, and the aligned label names.

        Returns
        -------
        tuple[list[pd.Series], list[str]]
            The list of all the timeseries and a list of the associated labels.
        """
        timeseries_labels = list(self.timeseries.keys())
        return [self.timeseries[timeseries_label][0] for timeseries_label in timeseries_labels], timeseries_labels

    def get_labels_all(self: TimeSeriesContainer) -> list[str]:
        """
        Get all the timeseries labels.

        Returns
        -------
        List
            Get the list of all timeseries labels.
        """
        return list(self.timeseries.keys())

    def get_units_all(self: TimeSeriesContainer) -> tuple[list[str], list[str]]:
        """
        Get the list of all units, and the associated labels.

        Returns
        -------
        list[str]
            The list containing all of the units and a list of the associated labels.
        """
        timeseries_labels = list(self.timeseries.keys())
        return [self.timeseries[timeseries_label][1] for timeseries_label in timeseries_labels], timeseries_labels


class AggregationStrategy(Enum):
    """The available aggregation types."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"


class AggregationPeriod(Enum):
    """The available aggregation periods."""

    Year = "mean"
    Quarter = "median"
    Month = "max"
