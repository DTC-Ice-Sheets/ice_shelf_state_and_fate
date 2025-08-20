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

    def __init__(self, aggregation: str):
        self.aggregation = aggregation


class AggregationPeriod(Enum):
    """The available aggregation periods."""

    YEAR = ("YS", "Y")
    QUARTER = ("QS", "Q")
    MONTH = ("MS", "M")
    DAY = ("d", "d")

    def __init__(self, period: str, time_period_shorthand: str):
        self.period = period
        self.time_period_shorthand = time_period_shorthand


class TimeSeriesPreprocesser:
    """Module for the manipulation of timeseries."""

    def __init__(
        self: TimeSeriesPreprocesser,
        timeseries: list[Timeseries],
        time_period: AggregationPeriod,
        aggregation: AggregationStrategy = AggregationStrategy.MEAN,
        process_timeseries: bool = True,
    ) -> None:
        """
        Initialise and create the necessary timeseries container.

        Parameters
        ----------
        timeseries : list[Timeseries]
            The timeseries as Timeseries objects.
        """
        self.timeseries = {}
        self.period = time_period.period
        self.time_period = time_period.time_period_shorthand
        self.aggregation = aggregation.aggregation
        self.add_timeseries(timeseries=timeseries)
        if process_timeseries:
            self.process_timeseries()

    def process_timeseries(
        self: TimeSeriesPreprocesser,
    ) -> None:
        """
        Process the timeseries for the covariate analysis. This first checks that all the timeseries intersect in time,
        it then resamples the timeseries to be the same frequency, trims the time series to be the same length and then
        interpolates any missing values using linear interpolation.
        """
        if not self._check_timeseries_intersection():
            raise ValueError("One or more of the specified time series don't intersect temporally")

        self._resample_timeseries()

        start_date, end_date = self._find_start_end_dates()

        self._trim_timeseries_date(start_date=start_date, end_date=end_date)

        self._interpolate_timeseries()

    def _check_timeseries_intersection(
        self: TimeSeriesPreprocesser,
    ) -> bool:
        """
        Determine whether all time series intersect in time by at least one time point. If both input parameters are
        None then it will check all of the timeseries in the container.
        """
        start_datetime, end_datetime = self._find_start_end_dates()

        return end_datetime >= start_datetime

    def _find_start_end_dates(
        self: TimeSeriesPreprocesser,
    ) -> tuple[pd.Timestamp | np.datetime64, pd.Timestamp | np.datetime64]:
        """
        Return the greatest common start and the smallest common end time for a set of time series. If both input
        parameters are None then it will check all of the timeseries in the container.

        Returns
        -------
        tuple[pd.Timestamp | np.datetime64, pd.Timestamp | np.datetime64]
            The start and end datetime. Note it is possible for the end time to be less than the start time in cases
            where the time series do not intersect in time
        """
        # Find the latest start datetime for all the input timeseries
        start_date = np.max(
            [np.min(self.timeseries[ts_label].measurements.index) for ts_label in self.timeseries.keys()]
        )
        # Find the earliest end datetime for all the input timeseries
        end_date = np.min([np.max(self.timeseries[ts_label].measurements.index) for ts_label in self.timeseries.keys()])

        return start_date, end_date

    def _trim_timeseries_date(
        self: TimeSeriesPreprocesser,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> None:
        """
        Trim a time series using a start and end date time. If both input parameters are None then it will check all of
        the timeseries in the container.

        Parameters
        ----------
        start_date : pd.Timestamp
            The start date to trim.
        end_date : pd.Timestamp
            The end date to trim.
        """
        for ts_label in self.timeseries.keys():
            self.timeseries[ts_label].measurements = self.timeseries[ts_label].measurements[
                (self.timeseries[ts_label].measurements.index >= start_date)
                & (self.timeseries[ts_label].measurements.index <= end_date)
            ]
            if self.timeseries[ts_label].measurement_uncertainties is not None:
                self.timeseries[ts_label].measurement_uncertainties = self.timeseries[
                    ts_label
                ].measurement_uncertainties[
                    (self.timeseries[ts_label].measurement_uncertainties.index >= start_date)
                    & (self.timeseries[ts_label].measurement_uncertainties.index <= end_date)
                ]

    def _resample_timeseries(
        self: TimeSeriesPreprocesser,
    ) -> None:
        """
        Resample the 1D timeseries into specific time periods (days, weeks, months, quarters, years).

        TODO: This method is to complex with the nested if statements. Not sure if there is a better way fo doing this
        though.
        """
        for ts_label in self.timeseries:
            timeseries_agg = deepcopy(self.timeseries[ts_label])
            timeseries_agg.measurements = (
                self.timeseries[ts_label]
                .measurements.groupby(
                    pd.PeriodIndex(self.timeseries[ts_label].measurements.index, freq=self.time_period)
                )
                .agg(self.aggregation)
            )
            if timeseries_agg.measurement_uncertainties is not None:
                timeseries_agg.measurement_uncertainties = (
                    self.timeseries[ts_label]
                    .measurement_uncertainties.groupby(
                        pd.PeriodIndex(
                            self.timeseries[ts_label].measurement_uncertainties.index,
                            freq=self.time_period,
                        )
                    )
                    .agg(self.aggregation)
                )
            timeseries_agg.measurements.index = timeseries_agg.measurements.index.to_timestamp()
            if timeseries_agg.measurement_uncertainties is not None:
                timeseries_agg.measurement_uncertainties.index = (
                    timeseries_agg.measurement_uncertainties.index.to_timestamp()
                )
            self.timeseries[ts_label] = timeseries_agg

    def _interpolate_timeseries(
        self: TimeSeriesPreprocesser,
    ) -> None:
        """Resamples any missing data and then linearly interpolates a timeseries according to a specified period."""
        for ts_label in self.timeseries.keys():
            timeseries_interp = deepcopy(self.timeseries[ts_label])
            # Generate the list of date idx
            date_idx = pd.date_range(
                start=timeseries_interp.measurements.index[0],
                end=timeseries_interp.measurements.index[-1],
                freq=self.period,
            )

            timeseries_interp.measurements = timeseries_interp.measurements.reindex(
                date_idx, fill_value=np.nan
            ).interpolate(method="linear")

            # Repeat if there are uncertainty measurements
            if timeseries_interp.measurement_uncertainties is not None:
                date_idx = pd.date_range(
                    start=timeseries_interp.measurement_uncertainties.index[0],
                    end=timeseries_interp.measurement_uncertainties.index[-1],
                    freq=self.period,
                )

                timeseries_interp.measurement_uncertainties = timeseries_interp.measurement_uncertainties.reindex(
                    date_idx, fill_value=np.nan
                ).interpolate(method="linear")

            self.timeseries[ts_label] = timeseries_interp

    def _infer_frequency(
        self: TimeSeriesPreprocesser,
        timeseries_label: str,
    ) -> str:
        """
        Infer the frequency between period index and return this as a string. Note only one of either the timeseries or
        label can be provided.

        Returns
        -------
        str
            The period as a string.
        """
        timeseries = self.timeseries[timeseries_label]

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
        self: TimeSeriesPreprocesser,
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

    def get_timeseries_all(self: TimeSeriesPreprocesser) -> list[Timeseries]:
        """
        Get all the timeseries, and the aligned label names.

        Returns
        -------
        list[Timeseries]
            The list of all the timeseries. and a list of the associated labels.
        """
        timeseries_labels = list(self.timeseries.keys())
        return [self.timeseries[timeseries_label] for timeseries_label in timeseries_labels]

    def get_labels_all(self: TimeSeriesPreprocesser) -> list[str]:
        """
        Get all the timeseries labels.

        Returns
        -------
        List
            Get the list of all timeseries labels.
        """
        return list(self.timeseries.keys())
