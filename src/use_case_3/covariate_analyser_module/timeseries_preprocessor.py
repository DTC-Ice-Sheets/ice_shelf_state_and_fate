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

    def __init__(self, aggregation: str) -> None:
        self.aggregation = aggregation


class AggregationPeriod(Enum):
    """The available aggregation periods."""

    YEAR = ("YS", "Y")
    QUARTER = ("QS", "Q")
    MONTH = ("MS", "M")
    DAY = ("d", "d")

    def __init__(self, period: str, time_period_shorthand: str) -> None:
        self.period = period
        self.time_period_shorthand = time_period_shorthand


class TimeSeriesPreprocessor:
    """Module for the manipulation of timeseries."""

    def __init__(
        self: TimeSeriesPreprocessor,
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
        self.aggregation_period = time_period
        self.aggregation = aggregation
        self.add_timeseries(timeseries=timeseries)
        self.start_date = None
        self.end_date = None
        if process_timeseries:
            self.process_timeseries()

    def process_timeseries(
        self: TimeSeriesPreprocessor,
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
        self: TimeSeriesPreprocessor,
    ) -> bool:
        """
        Determine whether all time series intersect in time by at least one time point. If both input parameters are
        None then it will check all of the timeseries in the container.
        """
        start_datetime, end_datetime = self._find_start_end_dates()

        return end_datetime >= start_datetime

    def _find_start_end_dates(
        self: TimeSeriesPreprocessor,
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
        start_date = np.max([np.min(ts.measurements.index) for ts in self.timeseries.values()])
        # Find the earliest end datetime for all the input timeseries
        end_date = np.min([np.max(ts.measurements.index) for ts in self.timeseries.values()])

        return start_date, end_date

    def _trim_timeseries_date(
        self: TimeSeriesPreprocessor,
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
        self: TimeSeriesPreprocessor,
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
                    pd.PeriodIndex(
                        self.timeseries[ts_label].measurements.index, freq=self.aggregation_period.time_period_shorthand
                    )
                )
                .agg(self.aggregation.aggregation)
            )
            if timeseries_agg.measurement_uncertainties is not None:
                timeseries_agg.measurement_uncertainties = (
                    self.timeseries[ts_label]
                    .measurement_uncertainties.groupby(
                        pd.PeriodIndex(
                            self.timeseries[ts_label].measurement_uncertainties.index,
                            freq=self.aggregation_period.time_period_shorthand,
                        )
                    )
                    .agg(self.aggregation.aggregation)
                )
            timeseries_agg.measurements.index = timeseries_agg.measurements.index.to_timestamp()
            if timeseries_agg.measurement_uncertainties is not None:
                timeseries_agg.measurement_uncertainties.index = (
                    timeseries_agg.measurement_uncertainties.index.to_timestamp()
                )
            self.timeseries[ts_label] = timeseries_agg

    def _interpolate_timeseries(
        self: TimeSeriesPreprocessor,
    ) -> None:
        """Resamples any missing data and then linearly interpolates a timeseries according to a specified period."""
        for ts_label in self.timeseries.keys():
            timeseries_interp = deepcopy(self.timeseries[ts_label])
            # Generate the list of date idx
            date_idx = pd.date_range(
                start=timeseries_interp.measurements.index[0],
                end=timeseries_interp.measurements.index[-1],
                freq=self.aggregation_period.period,
            )

            timeseries_interp.measurements = timeseries_interp.measurements.reindex(
                date_idx, fill_value=np.nan
            ).interpolate(method="linear")

            # Repeat if there are uncertainty measurements
            if timeseries_interp.measurement_uncertainties is not None:
                date_idx = pd.date_range(
                    start=timeseries_interp.measurement_uncertainties.index[0],
                    end=timeseries_interp.measurement_uncertainties.index[-1],
                    freq=self.aggregation_period.period,
                )

                timeseries_interp.measurement_uncertainties = timeseries_interp.measurement_uncertainties.reindex(
                    date_idx, fill_value=np.nan
                ).interpolate(method="linear")

            self.timeseries[ts_label] = timeseries_interp

    def add_timeseries(
        self: TimeSeriesPreprocessor,
        timeseries: list[Timeseries],
    ) -> None:
        """
        Set all the time series. It is required that labels and units are also provided for every timeseries.

        Parameters
        ----------
        timeseries : list[Timeseries]
            The timeseries as a dictionary where the key is a label in the list of timeseries labels.
        """
        for timeseries_single in timeseries:
            self.timeseries[timeseries_single.label] = timeseries_single
