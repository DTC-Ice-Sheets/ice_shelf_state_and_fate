"""Module for the manipulation of timeseries."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
)

Timestamp = pd.Timestamp | np.datetime64 | int


class TimeSeriesPreprocessor:
    """
    Module for the manipulation of timeseries.

    Parameters
    ----------
    timeseries : list[Timeseries]
            The list of timeseries objects.
    time_period : AggregationPeriod | None
        The aggregation period to apply. If None, the coarsest frequency from the input timeseries is used.
    aggregation : AggregationStrategy
        The aggregation type to apply, by default AggregationStrategy.MEAN
    process_timeseries : bool
        Whether to preprocess all of the input timeseries, by default True
    """

    def __init__(
        self: TimeSeriesPreprocessor,
        timeseries: list[Timeseries],
        time_period: AggregationPeriod | None = None,
        aggregation: AggregationStrategy = AggregationStrategy.MEAN,
        process_timeseries: bool = True,
    ) -> None:
        if time_period is None:
            # Take the largest period (i.e. lowest frequency) in the input timeseries
            max_days = 1
            for ts in timeseries:
                ts_max_days = ts.measurements.index.diff().max().ceil("D").days
                if ts_max_days > max_days:
                    max_days = ts_max_days
            if max_days >= 365:
                self.aggregation_period = AggregationPeriod.YEAR
            elif max_days >= 90:
                self.aggregation_period = AggregationPeriod.QUARTER
            elif max_days >= 28:
                self.aggregation_period = AggregationPeriod.MONTH
            else:
                self.aggregation_period = AggregationPeriod.DAY
        else:
            self.aggregation_period = time_period

        self.timeseries = {}
        self._timeseries_merge = []
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
        Process the timeseries for the covariate analysis.

        This first checks that all the timeseries intersect in time, it then resamples the timeseries to be the
        same frequency, trims the time series to be the same length and then interpolates any missing values
        using linear interpolation.
        """
        self._merge_timeseries()

        if not self._check_timeseries_intersection():
            raise ValueError("One or more of the specified time series don't intersect temporally")

        self._resample_timeseries()

        start_date, end_date = self.find_start_end_dates()

        self.trim_start_end_dates(start_date=start_date, end_date=end_date)

        self._interpolate_timeseries()

    def _check_timeseries_intersection(
        self: TimeSeriesPreprocessor,
    ) -> bool:
        """
        Determine whether all time series intersect in time by at least one time point.

        If both input parameters are None then it will check all of the timeseries in the container.
        """
        start_datetime, end_datetime = self.find_start_end_dates()

        return end_datetime >= start_datetime

    def find_start_end_dates(
        self: TimeSeriesPreprocessor,
    ) -> tuple[Timestamp, Timestamp]:
        """
        Return the greatest common start and the smallest common end time for a set of time series.

        If both input parameters are None then it will check all of the timeseries in the container.

        Returns
        -------
        tuple[Timestamp, Timestamp]
            The start and end datetime. Note it is possible for the end time to be less than the start time in cases
            where the time series do not intersect in time
        """
        # Find the latest start datetime for all the input timeseries
        start_date = np.max([np.min(ts.measurements.index) for ts in self.timeseries.values()])
        # Find the earliest end datetime for all the input timeseries
        end_date = np.min([np.max(ts.measurements.index) for ts in self.timeseries.values()])

        return start_date, end_date

    def trim_start_end_dates(
        self: TimeSeriesPreprocessor,
        start_date: Timestamp,
        end_date: Timestamp,
    ) -> None:
        """
        Trim a time series using a start and end date time.

        If both input parameters are None then it will check all of the timeseries in the container.

        Parameters
        ----------
        start_date : Timestamp
            The start date to trim.
        end_date : Timestamp
            The end date to trim.
        """
        for ts in self.timeseries.values():
            ts.measurements = ts.measurements[
                (ts.measurements.index >= start_date) & (ts.measurements.index <= end_date)
            ]
            if ts.measurement_uncertainties is not None:
                ts.measurement_uncertainties = ts.measurement_uncertainties[
                    (ts.measurement_uncertainties.index >= start_date)
                    & (ts.measurement_uncertainties.index <= end_date)
                ]

    def _merge_timeseries(self: TimeSeriesPreprocessor) -> None:
        """
        Merge time series that share the same label.

        Average matching series across columns after resampling.
        """
        for ts_label in self._timeseries_merge:
            ts_for_merging = []
            while len(self.timeseries[ts_label]) > 0:
                ts = self.timeseries[ts_label].pop(0)
                # display(self._resample_timeseries_agg(time_series=ts))
                ts_for_merging.append(self._resample_timeseries_agg(time_series=ts))
            ts_df = pd.concat([ts.measurements for ts in ts_for_merging], axis=1)
            self.timeseries[ts_label] = Timeseries(
                label=ts_label,
                measurements=ts_df.mean(axis=1),
                unit=ts_for_merging[0].unit,
                timeseries_type=ts_for_merging[0].timeseries_type,
            )

    def _resample_timeseries_agg(self: TimeSeriesPreprocessor, time_series: Timeseries) -> Timeseries:
        """
        Resample and aggregate a time series.

        Parameters
        ----------
        time_series : Timeseries
            Provide the source time series to resample.

        Returns
        -------
        Timeseries
            Return a copied series with measurements and
            uncertainties resampled to the configured period.
        """
        timeseries_agg = deepcopy(time_series)
        timeseries_agg.measurements = time_series.measurements.groupby(
            pd.PeriodIndex(
                time_series.measurements.index,
                freq=AggregationPeriod.pd_period_freq(self.aggregation_period),
            )
        ).agg(self.aggregation.aggregation)
        if timeseries_agg.measurement_uncertainties is not None:
            timeseries_agg.measurement_uncertainties = time_series.measurement_uncertainties.groupby(
                pd.PeriodIndex(
                    time_series.measurement_uncertainties.index,
                    freq=AggregationPeriod.pd_period_freq(self.aggregation_period),
                )
            ).agg(self.aggregation.aggregation)
        timeseries_agg.measurements.index = timeseries_agg.measurements.index.to_timestamp()
        if timeseries_agg.measurement_uncertainties is not None:
            timeseries_agg.measurement_uncertainties.index = (
                timeseries_agg.measurement_uncertainties.index.to_timestamp()
            )

        return timeseries_agg

    def _resample_timeseries(
        self: TimeSeriesPreprocessor,
    ) -> None:
        """
        Resample the 1D timeseries into specific time periods (days, weeks, months, quarters, years).

        TODO: This method is to complex with the nested if statements. Not sure if there is a better way fo doing this
        though.
        """
        for ts_label, ts in self.timeseries.items():
            self.timeseries[ts_label] = self._resample_timeseries_agg(time_series=ts)

    def _interpolate_timeseries(
        self: TimeSeriesPreprocessor,
    ) -> None:
        """Resamples any missing data and then linearly interpolates a timeseries according to a specified period."""
        for ts_label, ts in self.timeseries.items():
            timeseries_interp = deepcopy(ts)
            # Generate the list of date idx
            date_idx = pd.date_range(
                start=timeseries_interp.measurements.index[0],
                end=timeseries_interp.measurements.index[-1],
                freq=AggregationPeriod.pd_date_range_freq(self.aggregation_period),
            )

            timeseries_interp.measurements = timeseries_interp.measurements.reindex(
                date_idx, fill_value=np.nan
            ).interpolate(method="linear")

            # Repeat if there are uncertainty measurements
            if timeseries_interp.measurement_uncertainties is not None:
                date_idx = pd.date_range(
                    start=timeseries_interp.measurement_uncertainties.index[0],
                    end=timeseries_interp.measurement_uncertainties.index[-1],
                    freq=AggregationPeriod.pd_date_range_freq(self.aggregation_period),
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
        Set all the time series.

        It is required that labels and units are also provided for every timeseries.

        Parameters
        ----------
        timeseries : list[Timeseries]
            The timeseries as a dictionary where the key is a label in the list of timeseries labels.
        """
        for timeseries_single in timeseries:
            if timeseries_single.label not in self.timeseries:
                self.timeseries[timeseries_single.label] = timeseries_single
            else:
                if isinstance(self.timeseries[timeseries_single.label], list):
                    self.timeseries[timeseries_single.label].append(timeseries_single)
                else:
                    self.timeseries[timeseries_single.label] = [
                        self.timeseries[timeseries_single.label],
                        timeseries_single,
                    ]
                    self._timeseries_merge.append(timeseries_single.label)
