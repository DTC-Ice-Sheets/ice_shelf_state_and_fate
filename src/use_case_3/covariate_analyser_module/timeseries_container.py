"""Module for the manipulation of timeseries."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum

import numpy as np
import pandas as pd

from use_case_3.timeseries_module.timeseries import Timeseries


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


class TimeSeriesContainer:
    """Module for the manipulation of timeseries."""

    def __init__(
        self: TimeSeriesContainer,
        timeseries: list[Timeseries],
    ) -> None:
        """
        Initialise and create the necessary timeseries container.

        Parameters
        ----------
        timeseries : list[Timeseries]
            The timeseries as Timeseries objects.
        """
        self.timeseries = {}
        self.add_timeseries(timeseries=timeseries)

    def prepare_timeseries(
        self: TimeSeriesContainer,
        time_period: str,
        timeseries_labels: list[str] | None = None,
        aggregation: Enum = AggregationStrategy.MEAN,
        inplace: bool = True,
    ) -> list[Timeseries] | None:
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
        list[Timeseries] | None
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
            timeseries = self.get_timeseries_all()
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
                overwrite=True,
            )
        else:
            return timeseries

    def _combine_timeseries_from_labels_and_list(
        self: TimeSeriesContainer,
        timeseries_labels: list[str] | None = None,
        timeseries: list[Timeseries] | None = None,
    ) -> list[Timeseries]:
        """
        Combine lists of timeseries collected using either labels or directly as a passed list. If no list of labels or
        list of timeseries is passed, the default behavior is to return all of the timeseries in this container.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The list of timeseries, by default None
        timeseries : list[Timeseries] | None, optional
            The list of timeseries labels, by default None

        Returns
        -------
        list[Timeseries]
            The total list of timeseries.
        """
        if timeseries_labels is not None:
            timeseries_from_labels = self.get_timeseries(timeseries_labels=timeseries_labels)
        elif timeseries_labels is None and timeseries is None:
            timeseries_from_labels = self.get_timeseries_all()
        else:
            timeseries_from_labels = []

        if timeseries is not None:
            timeseries += timeseries_from_labels
        else:
            timeseries = timeseries_from_labels

        return timeseries

    def _check_timeseries_intersection(
        self: TimeSeriesContainer,
        timeseries_labels: list[str] | None = None,
        timeseries: list[Timeseries] | None = None,
    ) -> bool:
        """
        Determine whether all time series intersect in time by at least one time point. If both input parameters are
        None then it will check all of the timeseries in the container.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[Timeseries] | None, optional
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
        self: TimeSeriesContainer,
        timeseries_labels: list[str] | None = None,
        timeseries: list[Timeseries] | None = None,
    ) -> tuple[pd.Timestamp | np.datetime64, pd.Timestamp | np.datetime64]:
        """
        Return the greatest common start and the smallest common end time for a set of time series. If both input
        parameters are None then it will check all of the timeseries in the container.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[Timeseries] | None, optional
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
        start_date = np.max([np.min(series.measurements.index) for series in timeseries])
        # Find the earliest end datetime for all the input timeseries
        end_date = np.min([np.max(series.measurements.index) for series in timeseries])

        return start_date, end_date

    def _trim_timeseries_date(
        self: TimeSeriesContainer,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        timeseries_labels: list[str] | None = None,
        timeseries: list[Timeseries] | None = None,
    ) -> list[Timeseries]:
        """
        Trim a time series using a start and end date time. If both input parameters are None then it will check all of
        the timeseries in the container.

        Parameters
        ----------
        start_date : pd.Timestamp
            The start date to trim.
        end_date : pd.Timestamp
            The end date to trim.
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[Timeseries] | None, optional
            The list of time series, by default None

        Returns
        -------
        list[Timeseries]
            The trimmed timeseries with minimum time start_date and maximum time end_date
        """
        timeseries = self._combine_timeseries_from_labels_and_list(
            timeseries_labels=timeseries_labels, timeseries=timeseries
        )

        timeseries_trim = []
        for timeseries_single in timeseries:
            timeseries_single.measurements = timeseries_single.measurements[
                (timeseries_single.measurements.index >= start_date)
                & (timeseries_single.measurements.index <= end_date)
            ]
            if timeseries_single.measurement_uncertainties is not None:
                timeseries_single.measurement_uncertainties = timeseries_single.measurement_uncertainties[
                    (timeseries_single.measurement_uncertainties.index >= start_date)
                    & (timeseries_single.measurement_uncertainties.index <= end_date)
                ]

            timeseries_trim.append(timeseries_single)

        return timeseries_trim

    def _resample_timeseries(  # noqa C901
        self: TimeSeriesContainer,
        time_period: str,
        timeseries_labels: list[str] | None = None,
        timeseries: list[Timeseries] | None = None,
        aggregation: Enum = AggregationStrategy.MEAN,
    ) -> list[Timeseries]:
        """
        Resample the 1D timeseries into specific time periods (days, weeks, months, quarters, years).

        TODO: This method is to complex with the nested if statements. Not sure if there is a better way fo doing this
        though.

        Parameters
        ----------
        time_period : str
            The time period.
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[Timeseries] | None, optional
            The list of time series, by default None
        aggregation : str, optional
            The aggregation to be applied, by default "mean"

        Returns
        -------
        list[Timeseries]
            The aggregated time series.
        """
        timeseries = self._combine_timeseries_from_labels_and_list(
            timeseries_labels=timeseries_labels, timeseries=timeseries
        )

        timeseries_agg_list = []
        for timeseries_single in timeseries:
            timeseries_agg = deepcopy(timeseries_single)
            if aggregation == AggregationStrategy.MEAN:
                timeseries_agg.measurements = timeseries_single.measurements.groupby(
                    pd.PeriodIndex(timeseries_single.measurements.index, freq=time_period)
                ).mean()
                if timeseries_agg.measurement_uncertainties is not None:
                    timeseries_agg.measurement_uncertainties = timeseries_single.measurement_uncertainties.groupby(
                        pd.PeriodIndex(timeseries_single.measurement_uncertainties.index, freq=time_period)
                    ).mean()
            elif aggregation == AggregationStrategy.MEDIAN:
                timeseries_agg.measurements = timeseries_single.measurements.groupby(
                    pd.PeriodIndex(timeseries_single.measurements.index, freq=time_period)
                ).median()
                if timeseries_agg.measurement_uncertainties is not None:
                    timeseries_agg.measurement_uncertainties = timeseries_single.measurement_uncertainties.groupby(
                        pd.PeriodIndex(timeseries_single.measurement_uncertainties.index, freq=time_period)
                    ).median()
            elif aggregation == AggregationStrategy.MAX:
                timeseries_agg.measurements = timeseries_single.measurements.groupby(
                    pd.PeriodIndex(timeseries_single.measurements.index, freq=time_period)
                ).max()
                if timeseries_agg.measurement_uncertainties is not None:
                    timeseries_agg.measurement_uncertainties = timeseries_single.measurement_uncertainties.groupby(
                        pd.PeriodIndex(timeseries_single.measurement_uncertainties.index, freq=time_period)
                    ).max()
            elif aggregation == AggregationStrategy.MIN:
                timeseries_agg.measurements = timeseries_single.measurements.groupby(
                    pd.PeriodIndex(timeseries_single.measurements.index, freq=time_period)
                ).min()
                if timeseries_agg.measurement_uncertainties is not None:
                    timeseries_agg.measurement_uncertainties = timeseries_single.measurement_uncertainties.groupby(
                        pd.PeriodIndex(timeseries_single.measurement_uncertainties.index, freq=time_period)
                    ).min()
            timeseries_agg.measurements.index = timeseries_agg.measurements.index.to_timestamp()
            if timeseries_agg.measurement_uncertainties is not None:
                timeseries_agg.measurement_uncertainties.index = (
                    timeseries_agg.measurement_uncertainties.index.to_timestamp()
                )
            timeseries_agg_list.append(timeseries_agg)

        return timeseries_agg_list

    def _interpolate_timeseries(
        self: TimeSeriesContainer,
        period: str,
        timeseries_labels: list[str] | None = None,
        timeseries: list[Timeseries] | None = None,
    ) -> list[Timeseries]:
        """
        Resamples any missing data and then linearly interpolates a timeseries according to a specified period.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[Timeseries] | None, optional
            The list of time series, by default None
        period : str
            The period to resample the time series.

        Returns
        -------
        list[Timeseries]
            The resampled and interpolated timeseries.
        """
        timeseries = self._combine_timeseries_from_labels_and_list(
            timeseries_labels=timeseries_labels, timeseries=timeseries
        )

        timeseries_interpolated = []
        for timeseries_single in timeseries:
            timeseries_interp = deepcopy(timeseries_single)
            # Generate the list of date idx
            date_idx = pd.date_range(
                start=timeseries_interp.measurements.index[0],
                end=timeseries_interp.measurements.index[-1],
                freq=period,
            )

            timeseries_interp.measurements = timeseries_interp.measurements.reindex(
                date_idx, fill_value=np.nan
            ).interpolate(method="linear")

            # Repeat if there are uncertainty measurements
            if timeseries_interp.measurement_uncertainties is not None:
                date_idx = pd.date_range(
                    start=timeseries_interp.measurement_uncertainties.index[0],
                    end=timeseries_interp.measurement_uncertainties.index[-1],
                    freq=period,
                )

                timeseries_interp.measurement_uncertainties = timeseries_interp.measurement_uncertainties.reindex(
                    date_idx, fill_value=np.nan
                ).interpolate(method="linear")

            timeseries_interpolated.append(timeseries_interp)

        return timeseries_interpolated

    def _infer_frequency(
        self: TimeSeriesContainer,
        timeseries_label: str | None = None,
        timeseries: Timeseries | None = None,
    ) -> str:
        """
        Infer the frequency between period index and return this as a string. Note only one of either the timeseries or
        label can be provided.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            The labels of the timeseries to be checked, by default None
        timeseries : List[Timeseries] | None, optional
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

        timeseries = deepcopy(timeseries)
        if not isinstance(timeseries.measurements.index, pd.PeriodIndex):
            timeseries.measurements.index = timeseries.measurements.index.to_period()
        freqstr = timeseries.measurements.index.freq.freqstr
        if freqstr[0] == "M":
            freq = "Month"
        elif freqstr[0] == "Q":
            freq = "Quarter"
        elif freqstr[0] == "Y":
            freq = "Year"

        return freq

    def add_timeseries(
        self: TimeSeriesContainer,
        timeseries: list[Timeseries],
        overwrite: bool = False,
    ) -> None:
        """
        Set all the time series. It is required that labels and units are also provided for every timeseries.

        Parameters
        ----------
        timeseries : list[Timeseries]
            The timeseries as a dictionary where the key is a label in the list of timeseries labels.
        overwrite : bool, optional
            if set to True all the timeseries, labels and units will be overwritten. If False then these
            will only be added if no time series have been added already, by default False

        Raises
        ------
        ValueError
            Timeseries data already present and overwrite is set to False
        """
        for timeseries_single in timeseries:
            if timeseries_single.label in self.timeseries.keys():
                label_found = True
            else:
                label_found = False

            if not overwrite and label_found:
                raise ValueError("Timeseries is already added")
            else:
                self.timeseries[timeseries_single.label] = timeseries_single

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

    def get_timeseries(self: TimeSeriesContainer, timeseries_labels: list[str]) -> list[Timeseries]:
        """
        Get the specified timeseries.

        Parameters
        ----------
        label : list[str]
            The timeseries name.

        Returns
        -------
        list[Timeseries]
            The timeseries.

        Raises
        ------
        ValueError
            "There is no timeseries with this name"
        """
        timeseries = []
        for timeseries_label in timeseries_labels:
            if timeseries_label in self.timeseries.keys():
                timeseries.append(self.timeseries[timeseries_label])
            else:
                raise ValueError("There is no timeseries with this name")

        return timeseries

    def get_timeseries_all(self: TimeSeriesContainer) -> list[Timeseries]:
        """
        Get all the timeseries, and the aligned label names.

        Returns
        -------
        list[Timeseries]
            The list of all the timeseries. and a list of the associated labels.
        """
        timeseries_labels = list(self.timeseries.keys())
        return [self.timeseries[timeseries_label] for timeseries_label in timeseries_labels]

    def get_labels_all(self: TimeSeriesContainer) -> list[str]:
        """
        Get all the timeseries labels.

        Returns
        -------
        List
            Get the list of all timeseries labels.
        """
        return list(self.timeseries.keys())
