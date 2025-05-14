"""Module for undertaking the covariate analysis."""

from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RANSACRegressor


class CovariateAnalyser:
    """The covariate analysis module."""

    def __init__(self: Any) -> None:
        """Covariate analysis module."""
        pass

    def check_timeseries_intersection(self: Any, timeseries: List[pd.Series]) -> bool:
        """
        Determine whether all time series intersect in time by at least one time point.

        Parameters
        ----------
        timeseries : List[pd.Series]
            The list of time series.

        Returns
        -------
        bool
            True if all time series intersect with at least one time point.
        """
        start_datetime, end_datetime = self.find_start_end_dates(timeseries=timeseries)

        return end_datetime >= start_datetime

    def find_start_end_dates(
        self: Any, timeseries: List[pd.Series]
    ) -> Tuple[Union[pd.Timestamp, np.datetime64], Union[pd.Timestamp, np.datetime64]]:
        """
        Return the greatest common start and the smallest common end time for a set of time series.

        Parameters
        ----------
        timeseries : List[pd.Series]
            The list of time series.

        Returns
        -------
        Tuple[Union[pd.Timestamp, np.datetime64], Union[pd.Timestamp, np.datetime64]]
            The start and end datetime. Note it is possible for the end time to be less than the start time in cases
            where the time series do not intersect in time
        """
        # Find the latest start datetime for all the input timeseries
        start_date = np.max([np.min(series.index) for series in timeseries])
        # Find the earliest end datetime for all the input timeseries
        end_date = np.min([np.max(series.index) for series in timeseries])

        return start_date, end_date

    def trim_timeseries_date(
        self: Any, timeseries: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Series:
        """
        Trim a time series using a start and end date time.

        Parameters
        ----------
        timeseries : pd.end_date
            The time series.
        start_date : int
            The start date to trim.
        end_date : int
            The end date to trim

        Returns
        -------
        pd.DataFrame
            _description_
        """
        return timeseries[(timeseries.index >= start_date) & (timeseries.index <= end_date)]

    def resample_timeseries(self: Any, timeseries: pd.Series, time_period: str) -> pd.Series:
        """
        Resample a 1D time series into a specific time period (days, weeks, months, quarters, years).

        Parameters
        ----------
        timeseries : pd.Series
            The input time series with the date time as the index.

        Returns
        -------
        pd.Series
            The aggregated time series.
        """
        # return timeseries.groupby(pd.PeriodIndex(timeseries.index, freq=time_period)).mean()
        data_agg = timeseries.groupby(pd.PeriodIndex(timeseries.index, freq=time_period)).mean()
        data_agg.index = data_agg.index.to_timestamp()
        return data_agg

    def interpolate_timeseries(self: Any, timeseries: pd.Series, period: str) -> pd.Series:
        """
        Resamples any missing data and then linearly interpolates a timeseries according to a specified period.

        Parameters
        ----------
        timeseries : pd.Series
            The specified timeseries. This must be a pd.Series with a date time index.
        period : str
            The period to resample the time series.

        Returns
        -------
        pd.Series
            The resampled and interpolated time series.
        """
        # Generate the list of date idx
        date_idx = pd.date_range(start=timeseries.index[0], end=timeseries.index[-1], freq=period)

        timeseries = timeseries.reindex(date_idx, fill_value=np.nan)
        return timeseries.interpolate(method="linear")

    def integrity_check_timeseries(self: Any, timeseries: List[pd.Series]) -> None:
        """
        Check the integrity of the time series.

        Parameters
        ----------
        time_series : List[pd.Series]
            The set of time series to be integrity checked.

        Raises
        ------
        ValueError
            The time series have different lengths.
        ValueError
            The time series have different lengths.
        """
        # Check the series have the same length
        n = timeseries[0].shape[0]
        if all(x.shape[0] == n for x in timeseries):
            pass
        else:
            raise ValueError("The time series have different lengths")

        # Check the series have the same time index
        idx = timeseries[0].index
        if all(idx.identical(series.index) for series in timeseries):
            pass
        else:
            raise ValueError("The time series have different indexes")

    def correlation_coefficient(self: Any, time_series_1: pd.Series, time_series_2: pd.Series) -> Tuple[float, float]:
        """
        Calculate the Pearson Correlation Coefficient between two time series.

        Parameters
        ----------
        time_series_1 : pd.Series
            The first time series.
        time_series_2 : pd.Series
            The second time series.

        Returns
        -------
        Tuple[float, float]
            The Pearson correlation coefficient and associated p-value.
        """
        corr = pearsonr(time_series_1, time_series_2)

        return corr.statistic, corr.pvalue

    def covariance_normalised(self: Any, time_series: List[pd.Series]) -> pd.DataFrame:
        """
        Calculate the normalised covariance matrix.

        Parameters
        ----------
        time_series : List[pd.Series]
            The list of time series.

        Returns
        -------
        pd.DataFrame
            The covariance matrix.
        """
        df = pd.concat(time_series, axis=1)
        return pd.DataFrame(np.corrcoef(df.T), columns=df.columns, index=df.columns)

    def crosscorrelation(
        self: Any, time_series_1: pd.Series, time_series_2: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the cross-correlation between two time series.

        Parameters
        ----------
        time_series_1 : pd.Series
            The first time series.
        time_series_2 : pd.Series
            The second time series.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The cross-correlation with assoicated lags.
        """
        time_series_1 = (time_series_1 - time_series_1.mean()) / np.std(time_series_1)
        time_series_2 = (time_series_2 - time_series_2.mean()) / np.std(time_series_2)

        return correlate(time_series_1, time_series_2, "full") / min(
            time_series_1.shape[0], time_series_2.shape[0]
        ), correlation_lags(time_series_1.shape[0], time_series_2.shape[0])

    def max_crosscorrelation(
        self: Any, time_series_1: pd.Series, time_series_2: pd.Series
    ) -> Tuple[np.float64, np.int64]:
        """
        Find the maximum cross-correlation and associated lag.

        Parameters
        ----------
        time_series_1 : pd.Series
            The first time series.
        time_series_2 : pd.Series
            The second time series.

        Returns
        -------
        Tuple[float, int]
            The maximum cross-correlation and associated lag.
        """
        corr, corr_lags = self.crosscorrelation(time_series_1=time_series_1, time_series_2=time_series_2)

        return corr[np.argmax(np.abs(corr))], corr_lags[np.argmax(np.abs(corr))]

    def temporal_offset(
        self: Any, time_series_1: pd.Series, time_series_2: pd.Series, metric: str = "mean"
    ) -> Tuple[Union[float, int, np.float64, np.int64], Union[np.ndarray, int, np.int64, None]]:
        """
        Calculate a temporal offset. This is the difference between two series. This can be done using various
        statistics such as the mean, median, maximum and minimum.

        Parameters
        ----------
        time_series_1 : pd.Series
            The first time series.
        time_series_2 : pd.Series
            The second time series.
        metric : str, optional
            The statistic to quantify the difference between the two time series, by default "mean".

        Returns
        -------
        Tuple[float, Union[int, None]]
            The temporal offset and if applicable the associated index.
        """
        diff = time_series_1 - time_series_2

        if metric == "mean":
            return diff.mean(), None
        elif metric == "median":
            return diff.median(), None
        elif metric == "max":
            return diff.max(), diff.argmax()
        elif metric == "min":
            return diff.min(), diff.argmin()

    def temporal_slope(
        self: Any, time_series_1: pd.Series, time_series_2: pd.Series, plot: bool = False
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate the temporal slope of two time series.

        Parameters
        ----------
        time_series_1 : pd.Series
            The first time series.
        time_series_2 : pd.Series
            The second time series.
        plot : bool, optional
            True if you want a plot of the model fits, by default False

        Returns
        -------
        Tuple[float, float, float, float, float]
            _description_
        """
        reg_1 = RANSACRegressor(random_state=0).fit(np.arange(0, time_series_1.shape[0]).reshape(-1, 1), time_series_1)
        reg_2 = RANSACRegressor(random_state=0).fit(np.arange(0, time_series_2.shape[0]).reshape(-1, 1), time_series_2)

        if plot:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].plot(np.arange(0, time_series_1.shape[0]).reshape(-1, 1), time_series_1, ".k")
            axes[0].plot(
                [0, time_series_1.shape[0]],
                [
                    reg_1.estimator_.intercept_,
                    (reg_1.estimator_.coef_[0] * time_series_1.shape[0]) + reg_1.estimator_.intercept_,
                ],
                "--k",
            )
            axes[0].grid()
            axes[1].plot(np.arange(0, time_series_2.shape[0]).reshape(-1, 1), time_series_2, ".k")
            axes[1].plot(
                [0, time_series_1.shape[0]],
                [
                    reg_2.estimator_.intercept_,
                    (reg_2.estimator_.coef_[0] * time_series_1.shape[0]) + reg_2.estimator_.intercept_,
                ],
                "--k",
            )
            axes[1].grid()
            plt.show()

        return (
            reg_1.estimator_.coef_[0] / reg_2.estimator_.coef_[0],
            reg_1.estimator_.coef_[0],
            reg_1.estimator_.intercept_,
            reg_2.estimator_.coef_[0],
            reg_2.estimator_.intercept_,
        )

    def jointplot(self, time_series_1: pd.Series, time_series_2: pd.Series) -> None:
        """
        Visualise a joint plot between two time series.

        Parameters
        ----------
        time_series_1 : pd.Series
            The first time series.
        time_series_2 : pd.Series
            The second time series.
        """
        reg_1 = LinearRegression().fit(time_series_1.values.reshape(-1, 1), time_series_2.values.reshape(-1, 1))

        joint = sns.jointplot(x=time_series_1, y=time_series_2)
        joint.ax_joint.plot(
            joint.ax_joint.get_xlim(),
            [
                (joint.ax_joint.get_xlim()[0] * reg_1.coef_[0]) + reg_1.intercept_,
                (joint.ax_joint.get_xlim()[1] * reg_1.coef_[0]) + reg_1.intercept_,
            ],
            color="grey",
            linestyle="--",
        )
        plt.grid()
        plt.show()

    # def jointplot_grid(self: Any, timeseries)
