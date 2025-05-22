"""Module for undertaking the covariate analysis."""

import textwrap
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RANSACRegressor


class CovariateAnalyser:
    """The covariate analysis module."""

    def __init__(
        self: Any,
        timeseries: Dict[str, pd.Series],
        timeseries_labels: List[str],
        timeseries_units: Dict[str, str],
        alpha: float = 0.05,
    ) -> None:
        """Covariate analysis module."""
        self.alpha = alpha
        self.timeseries = {}
        self.timeseries_labels = []
        self.timeseries_units = {}
        self.set_timeseries(
            timeseries=timeseries, timeseries_labels=timeseries_labels, timeseries_units=timeseries_units
        )

    def set_timeseries(
        self: Any,
        timeseries: Dict[str, pd.Series],
        timeseries_labels: List[str],
        timeseries_units: Dict[str, str],
        overwrite: bool = False,
    ) -> None:
        """
        Set all the time series. It is required that labels and units are also provided for every timeseries.

        Parameters
        ----------
        timeseries : Dict[str, pd.Series]
            The timeseries as a dictionary where the key is a label in the list of timeseries labels.
        timeseries_labels : List[str]
            The timeseries labels. These are the keys for the timeseries and associated units.
        timeseries_units : Dict[str, str]
            The timeseries units. If a time series is unit-less then use an empty string.
        overwrite : bool, optional
            if set to True all the timeseries, labels and units will be overwritten. If False then these
            will only be added if no time series have been added already, by default False

        Raises
        ------
        ValueError
            Timeseries data already present and overwrite is set to False
        """
        if len(self.timeseries_labels) > 0:
            data_present = True
        else:
            data_present = False

        if data_present and not overwrite:
            raise ValueError("Timeseries data already present and overwrite is set to False")
        else:
            self.timeseries_labels = timeseries_labels
            self.timeseries = timeseries
            self.timeseries_units = timeseries_units

    def add_timeseries(
        self: Any, timeseries: pd.Series, timeseries_label: str, timeseries_unit: str = "", overwrite: bool = False
    ) -> None:
        """
        Add a single timeseries.

        Parameters
        ----------
        timeseries : pd.Series
            The timeseries.
        label : str
            The timeseries label.
        timeseries_unit : str
            the timeseries unit.
        overwrite : bool, optional
            If True the timeseries will be overwritten, if False the timeseries will only be added if not
            present, by default False

        Raises
        ------
        ValueError
            "Timeseries is already added"
        """
        if timeseries_label in self.timeseries_labels:
            label_found = True
        else:
            label_found = False

        if overwrite and label_found:
            raise ValueError("Timeseries is already added")
        else:
            self.timeseries_labels.append(timeseries_label)
            self.timeseries[timeseries_label] = timeseries
            self.timeseries_units[timeseries_label] = timeseries_unit

    def remove_timeseries(self, timeseries_label: str) -> None:
        """
        Remove the specified timeseries.

        Parameters
        ----------
        label : str
            The time series label.

        Raises
        ------
        ValueError
            "There is no timeseries with this name"
        """
        if timeseries_label in self.timeseries_labels:
            del self.timeseries[timeseries_label]
            del self.timeseries_units[timeseries_label]
            self.timeseries_labels.remove(timeseries_label)
        else:
            raise ValueError("There is no timeseries with this name")

    def get_timeseries(self: Any, timeseries_label: str) -> pd.Series:
        """
        Get the specified timeseries.

        Parameters
        ----------
        label : str
            The timeseries name.

        Returns
        -------
        pd.Series
            The timeseries.

        Raises
        ------
        ValueError
            "There is no timeseries with this name"
        """
        if timeseries_label in self.timeseries_labels:
            return self.timeseries[timeseries_label]
        else:
            raise ValueError("There is no timeseries with this name")

    def get_units(self: Any, timeseries_label: str) -> str:
        """
        Get the unit associated with the timeseries.

        Parameters
        ----------
        label : str
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
        if timeseries_label in self.timeseries_labels:
            return self.timeseries_units[timeseries_label]
        else:
            raise ValueError("There is no timeseries with this name")

    def get_timeseries_all(self: Any) -> Dict:
        """
        Get all the timeseries.

        Returns
        -------
        Dict
            The dictionary of the timeseries.
        """
        return self.timeseries

    def get_labels_all(self: Any) -> List:
        """
        Get all the timeseries labels.

        Returns
        -------
        List
            Get the list of all timeseries.
        """
        return self.timeseries_labels

    def get_units_all(self: Any) -> Dict:
        """
        Get the list of all units.

        Returns
        -------
        Dict
            The dictionary containing all of the units.
        """
        return self.timeseries_units

    def prepare_timeseries(
        self: Any,
        time_period: str,
        timeseries_labels: Union[None, List[str]] = None,
        integrity_check: bool = True,
        aggregation: str = "mean",
    ) -> bool:
        """
        Prepare the timeseries for the covariate analysis.

        Parameters
        ----------
        time_period : str
            The time period.
        timeseries_labels : Union[None, List[str]], optional
            The labels of the timeseries to be checked, by default None
        integrity_check : bool, optional
            Whether the time series will be integrity checked after processing, by default True
        aggregation : str, optional
            The aggregation to be applied, by default "mean"

        Returns
        -------
        bool
            _description_
        """
        if time_period == "M":
            period = "MS"
        elif time_period == "Q":
            period = "QS"
        elif time_period == "Y":
            period = "YS"
        elif time_period == "d":
            period = "d"

        if timeseries_labels is None or len(timeseries_labels) == 0:
            timeseries_labels = deepcopy(self.timeseries_labels)

        for label in timeseries_labels:
            self.timeseries[label] = self.resample_timeseries(
                timeseries_label=label, time_period=time_period, aggregation=aggregation
            )

        start_date, end_date = self.find_start_end_dates(timeseries_labels)

        for label in timeseries_labels:
            self.timeseries[label] = self.trim_timeseries_date(
                timeseries_label=label, start_date=start_date, end_date=end_date
            )

        for label in timeseries_labels:
            self.timeseries[label] = self.interpolate_timeseries(label, period=period)

        if integrity_check:
            self.integrity_check_timeseries(timeseries_labels)

    def check_timeseries_intersection(self: Any, timeseries_labels: List[str]) -> bool:
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
        start_datetime, end_datetime = self.find_start_end_dates(timeseries_labels=timeseries_labels)

        return end_datetime >= start_datetime

    def find_start_end_dates(
        self: Any, timeseries_labels: Union[List[str], None] = None, timeseries: Union[List[pd.Series], None] = None
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
        if timeseries_labels is not None and timeseries is not None:
            timeseries = [self.timeseries[label] for label in timeseries_labels] + timeseries
        elif timeseries_labels is not None and timeseries is None:
            timeseries = [self.timeseries[label] for label in timeseries_labels]
        elif timeseries_labels is None and timeseries is not None:
            pass  # timeseries is already a list of timeseries, leaving this elif to be explicit and to make it easier
            # to follow.
        else:
            raise ValueError("Both the list of labels and timeseries can't be None")
        # Find the latest start datetime for all the input timeseries
        start_date = np.max([np.min(series.index) for series in timeseries])
        # Find the earliest end datetime for all the input timeseries
        end_date = np.min([np.max(series.index) for series in timeseries])

        return start_date, end_date

    def trim_timeseries_date(
        self: Any,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        timeseries_label: Union[str, None] = None,
        timeseries: Union[pd.Series, None] = None,
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
        if timeseries_label is not None and timeseries is not None:
            raise ValueError("Only a label or timeseries can be passed, please set one to None")
        elif timeseries_label is not None:
            timeseries = self.timeseries[timeseries_label]
        elif timeseries is not None:
            pass  # timeseries is already a pd.Series, leaving this elif to be explicit and to make it easier to
            # follow.
        else:
            raise ValueError("Both the label and timeseries can't be None")
        return timeseries[(timeseries.index >= start_date) & (timeseries.index <= end_date)]

    def resample_timeseries(self: Any, timeseries_label: str, time_period: str, aggregation: str = "mean") -> pd.Series:
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
        if aggregation == "mean":
            data_agg = (
                self.timeseries[timeseries_label]
                .groupby(pd.PeriodIndex(self.timeseries[timeseries_label].index, freq=time_period))
                .mean()
            )
        elif aggregation == "median":
            data_agg = (
                self.timeseries[timeseries_label]
                .groupby(pd.PeriodIndex(self.timeseries[timeseries_label].index, freq=time_period))
                .median()
            )
        elif aggregation == "max":
            data_agg = (
                self.timeseries[timeseries_label]
                .groupby(pd.PeriodIndex(self.timeseries[timeseries_label].index, freq=time_period))
                .max()
            )
        elif aggregation == "min":
            data_agg = (
                self.timeseries[timeseries_label]
                .groupby(pd.PeriodIndex(self.timeseries[timeseries_label].index, freq=time_period))
                .min()
            )
        data_agg.index = data_agg.index.to_timestamp()
        return data_agg

    def interpolate_timeseries(self: Any, timeseries_label: str, period: str) -> pd.Series:
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
        date_idx = pd.date_range(
            start=self.timeseries[timeseries_label].index[0],
            end=self.timeseries[timeseries_label].index[-1],
            freq=period,
        )

        timeseries = self.timeseries[timeseries_label].reindex(date_idx, fill_value=np.nan)
        return timeseries.interpolate(method="linear")

    def integrity_check_timeseries(self: Any, timeseries_labels: List[str]) -> None:
        """
        Check the integrity of the time series.

        Parameters
        ----------
        time_series : List[str]
            The set of time series to be integrity checked.

        Raises
        ------
        ValueError
            The time series have different lengths.
        ValueError
            The time series have different lengths.
        """
        # Check the series have the same length
        n = self.timeseries[timeseries_labels[0]].shape[0]
        if all(self.timeseries[label].shape[0] == n for label in timeseries_labels):
            pass
        else:
            raise ValueError("The time series have different lengths")

        # Check the series have the same time index
        idx = self.timeseries[timeseries_labels[0]].index
        if all(idx.identical(self.timeseries[label].index) for label in timeseries_labels):
            pass
        else:
            raise ValueError("The time series have different indexes")

    def correlation_coefficient_label(
        self: Any, timeseries_label_1: str, timeseries_label_2: str
    ) -> Tuple[float, float]:
        """
        Calculate the Pearson Correlation Coefficient between two time series.

        Parameters
        ----------
        timeseries_label_1 : str
            The label of the first time series.
        timeseries_label_2 : str
            The label of the second time series.

        Returns
        -------
        Tuple[float, float]
            The Pearson correlation coefficient and associated p-value.
        """
        corr, pvalue = self.correlation_coefficient(
            self.timeseries[timeseries_label_1], self.timeseries[timeseries_label_2]
        )

        return corr, pvalue

    def correlation_coefficient(self: Any, timeseries_1: pd.Series, timeseries_2: pd.Series) -> Tuple[float, float]:
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
        corr = pearsonr(timeseries_1, timeseries_2)

        return corr.statistic, corr.pvalue

    def covariance(self: Any, timeseries_labels: List[str], normalisation: Union[None, str] = None) -> pd.DataFrame:
        """
        Calculate the normalised covariance matrix.

        Parameters
        ----------
        time_series : List[pd.Series]
            The list of time series.
        normalisation : normalisation: Union[None, str], optional
            The type of normalisation applied, this can be None, mean_centered and correlation coefficient.

        Returns
        -------
        pd.DataFrame
            The covariance matrix.
        """
        if normalisation == "mean_centered":
            timeseries = [self.timeseries[label] - self.timeseries[label].mean() for label in timeseries_labels]
        else:
            timeseries = [self.timeseries[label] for label in timeseries_labels]
        df = pd.concat(timeseries, axis=1)
        if normalisation == "corrcoef":
            return pd.DataFrame(np.corrcoef(df.T), columns=df.columns, index=df.columns)
        else:
            return pd.DataFrame(np.cov(df.T), columns=df.columns, index=df.columns)

    def crosscorrelation(self: Any, timeseries_label_1: str, timeseries_label_2: str) -> Tuple[np.ndarray, np.ndarray]:
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
        time_series_1 = (self.timeseries[timeseries_label_1] - self.timeseries[timeseries_label_1].mean()) / np.std(
            self.timeseries[timeseries_label_1]
        )
        time_series_2 = (self.timeseries[timeseries_label_2] - self.timeseries[timeseries_label_2].mean()) / np.std(
            self.timeseries[timeseries_label_2]
        )

        return correlate(time_series_1, time_series_2, "full") / min(
            time_series_1.shape[0], time_series_2.shape[0]
        ), correlation_lags(time_series_1.shape[0], time_series_2.shape[0])

    def max_crosscorrelation(
        self: Any, timeseries_label_1: str, timeseries_label_2: str, max_lag: Union[int, np.int64, None] = None
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
        corr, corr_lags = self.crosscorrelation(
            timeseries_label_1=timeseries_label_1, timeseries_label_2=timeseries_label_2
        )

        if max_lag is None:
            max_lag = max(corr_lags)

        corr = [corr[idx] for idx in range(len(corr_lags)) if np.abs(corr_lags[idx]) <= max_lag]
        corr_lags = [corr_lags[idx] for idx in range(len(corr_lags)) if np.abs(corr_lags[idx]) <= max_lag]

        return corr[np.argmax(np.abs(corr))], corr_lags[np.argmax(np.abs(corr))]

    def infer_frequency(
        self: Any, timeseries_label: Union[str, None] = None, timeseries: Union[pd.Series, None] = None
    ) -> str:
        """
        Infer the frequency between period index and return this as a string. Note only one of the timeseries or label
        can be provided.

        Parameters
        ----------
        timeseries_label : Union[str, None], optional
            The timeseries label, by default None
        timeseries : Union[pd.Series, None], optional
            The timeseries, by default None

        Returns
        -------
        str
            The period as a string.
        """
        if timeseries is None and timeseries_label is not None:
            timeseries = self.timeseries[timeseries_label]

        timeseries = timeseries.copy()
        if not isinstance(timeseries.index, pd.PeriodIndex):
            # if type(timeseries.index) != pd.PeriodIndex:
            timeseries.index = timeseries.index.to_period()
        freqstr = timeseries.index.freq.freqstr
        if freqstr[0] == "M":
            freq = "Month"
        elif freqstr[0] == "Q":
            freq = "Quarter"
        elif freqstr[0] == "Y":
            freq = "Year"

        return freq

    def align_timeseries(
        self: Any, timeseries_label_1: str, timeseries_label_2: str, max_lag: Union[int, None] = None
    ) -> Tuple[pd.Series, pd.Series, str, Union[int, np.int64]]:
        """
        Align two timeseries according to the maximum cross correlation.

        Parameters
        ----------
        timeseries_label_1 : str
            The label for the first timeseries.
        timeseries_label_2 : str
            The label for the second timeseries.
        max_lag : Union[int, None], optional
            The maximum lag to consider, not this will apply this maximum lag in both the positive and
            the negative directions, by default None

        Returns
        -------
        Tuple[pd.Series, pd.Series, str, Union[int, np.int64]]
            The time aligned series according to the maximum cross correlation.
        """
        _, corr_lag = self.max_crosscorrelation(
            timeseries_label_1=timeseries_label_1,
            timeseries_label_2=timeseries_label_2,
            max_lag=max_lag,
        )

        timeseries_1 = self.get_timeseries(timeseries_label_1).copy()
        timeseries_2 = self.get_timeseries(timeseries_label_2).copy()

        freq = self.infer_frequency(timeseries=timeseries_1)

        if freq == "Month":
            timeseries_2.index = timeseries_2.index + pd.offsets.MonthEnd(corr_lag)
        elif freq == "Quarter":
            timeseries_2.index = timeseries_2.index + pd.offsets.QuarterEnd(corr_lag)
        elif freq == "Year":
            timeseries_2.index = timeseries_2.index + pd.offsets.YearEnd(corr_lag)

        start_date, end_date = self.find_start_end_dates(timeseries=[timeseries_1, timeseries_2])

        return (
            self.trim_timeseries_date(timeseries=timeseries_1, start_date=start_date, end_date=end_date),
            self.trim_timeseries_date(timeseries=timeseries_2, start_date=start_date, end_date=end_date),
            freq,
            corr_lag,
        )

    def temporal_offset(
        self: Any, timeseries_label_1: str, timeseries_label_2: str, metric: str = "mean"
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
        diff = self.timeseries[timeseries_label_1] - self.timeseries[timeseries_label_2]

        if metric == "mean":
            return diff.mean(), None
        elif metric == "median":
            return diff.median(), None
        elif metric == "max":
            return diff.max(), diff.argmax()
        elif metric == "min":
            return diff.min(), diff.argmin()

    def temporal_slope(
        self: Any, timeseries_label_1: str, timeseries_label_2: str, plot: bool = False
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
        reg_1 = RANSACRegressor(random_state=0).fit(
            np.arange(0, self.timeseries[timeseries_label_1].shape[0]).reshape(-1, 1),
            self.timeseries[timeseries_label_1],
        )
        reg_2 = RANSACRegressor(random_state=0).fit(
            np.arange(0, self.timeseries[timeseries_label_2].shape[0]).reshape(-1, 1),
            self.timeseries[timeseries_label_2],
        )

        if plot:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].plot(
                np.arange(0, self.timeseries[timeseries_label_1].shape[0]).reshape(-1, 1),
                self.timeseries[timeseries_label_1],
                ".k",
            )
            axes[0].plot(
                [0, self.timeseries[timeseries_label_1].shape[0]],
                [
                    reg_1.estimator_.intercept_,
                    (reg_1.estimator_.coef_[0] * self.timeseries[timeseries_label_1].shape[0])
                    + reg_1.estimator_.intercept_,
                ],
                "--k",
            )
            axes[0].grid()
            axes[1].plot(
                np.arange(0, self.timeseries[timeseries_label_2].shape[0]).reshape(-1, 1),
                self.timeseries[timeseries_label_2],
                ".k",
            )
            axes[1].plot(
                [0, self.timeseries[timeseries_label_2].shape[0]],
                [
                    reg_2.estimator_.intercept_,
                    (reg_2.estimator_.coef_[0] * self.timeseries[timeseries_label_2].shape[0])
                    + reg_2.estimator_.intercept_,
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

    def jointplot(self, timeseries_label_1: str, timeseries_label_2: str) -> None:
        """
        Visualise a joint plot between two time series.

        Parameters
        ----------
        time_series_1 : pd.Series
            The first time series.
        time_series_2 : pd.Series
            The second time series.
        """
        reg_1 = LinearRegression().fit(
            self.timeseries[timeseries_label_1].values.reshape(-1, 1),
            self.timeseries[timeseries_label_2].values.reshape(-1, 1),
        )

        joint = sns.jointplot(x=self.timeseries[timeseries_label_1], y=self.timeseries[timeseries_label_2])
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

    def jointplot_grid(self: Any, timeseries_labels: List[str], crosscorr: bool = False) -> None:
        """
        Plot a gridded jointplot for the given time series. Note that you can pass crosscorr True if the
        maximum cross-correlation is desired.

        Parameters
        ----------
        timeseries_labels : List[str]
            The timeseries labels to be plotted.
        crosscorr : bool, optional
            Whether the pearson correlation coefficient (False) or the maximum cross correlation should be used (True),
            by default False
        """
        timeseries = [self.timeseries[label] for label in timeseries_labels]

        width_ratios = [3] * len(timeseries) + [1]
        height_ratios = [3] * len(timeseries) + [1]

        f, axes = plt.subplots(
            len(height_ratios),
            len(width_ratios),
            gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
            figsize=(10, 10),
            sharex="col",
            sharey="row",
        )

        axes[len(timeseries), len(timeseries)].axis("off")

        for row in range(len(timeseries)):
            for col in range(len(timeseries)):
                if row != col:
                    if crosscorr:
                        timeseries_col, timeseries_row, freq, lag = self.align_timeseries(
                            timeseries_labels[col], timeseries_labels[row]
                        )
                    else:
                        timeseries_col = timeseries[col].copy()
                        timeseries_row = timeseries[row].copy()
                        lag = 0
                        freq = self.infer_frequency(timeseries=timeseries_col)

                    if np.abs(lag) != 1:
                        freq = freq + "s"
                    corr = self.correlation_coefficient(timeseries_col, timeseries_row)
                    sns.regplot(x=timeseries_col, y=timeseries_row, ax=axes[row, col])
                    axes[row, col].set_xlabel("")
                    axes[row, col].set_ylabel("")
                    axes[row, col].grid()
                    title = f"corr:{corr[0]:.2f}, p:{corr[1]:.2f}"
                    y_offset = 0.85
                    if crosscorr:
                        title += f", \n lag:{lag} {freq}"
                        y_offset = 0.75
                    axes[row, col].set_title(title, y=y_offset)
                    if corr[1] > self.alpha:
                        axes[row, col].set_facecolor("mistyrose")
                else:
                    axes[row, col].set_facecolor("lightgrey")

            # plot the histograms
            sns.histplot(data=timeseries[row].to_frame(), y=timeseries[row].to_frame().columns[0], ax=axes[row, -1])
            axes[row, -1].spines["top"].set_visible(False)
            axes[row, -1].spines["right"].set_visible(False)
            axes[row, -1].spines["bottom"].set_visible(False)
            axes[row, -1].set_xlabel("")
            axes[row, -1].get_xaxis().set_ticks([])

            sns.histplot(data=timeseries[row].to_frame(), x=timeseries[row].to_frame().columns[0], ax=axes[-1, row])
            axes[-1, row].spines["top"].set_visible(False)
            axes[-1, row].spines["right"].set_visible(False)
            axes[row, -1].spines["bottom"].set_visible(False)
            axes[-1, row].spines["left"].set_visible(False)
            axes[-1, row].set_ylabel("")
            axes[-1, row].get_yaxis().set_ticks([])

            # Add the correct axis labels
            label = textwrap.fill(
                " ".join([timeseries_labels[row], "(" + self.timeseries_units[timeseries_labels[row]] + ")"]),
                width=25,
                break_long_words=True,
            )
            axes[row, 0].set_ylabel(label)
            axes[-1, row].set_xlabel(label)

        plt.tight_layout()
        plt.plot()

    def covariate_analysis(self: Any, timeseries_labels):  # noqa
        self.jointplot_grid(timeseries_labels=timeseries_labels)
