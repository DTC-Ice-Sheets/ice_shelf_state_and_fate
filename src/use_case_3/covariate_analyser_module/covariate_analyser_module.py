"""Module for undertaking the covariate analysis."""

from typing import Any, Tuple, Union

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
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        """
        pass

    def trim_timeseries_date(self: Any, timeseries: pd.DataFrame, start_date: int, end_date: int) -> pd.DataFrame:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        timeseries : pd.DataFrame
            _description_
        start_date : int
            _description_
        end_date : int
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        return timeseries[(timeseries["datetime"] >= start_date) & (timeseries["datetime"] <= end_date)]

    def resample_timeseries(
        self: Any, timeseries: pd.DataFrame, time_period: str, target_column: str, date_column: str
    ) -> pd.DataFrame:
        """
        Resample a 1D time series into a specific time period (days, weeks, months, quarters, years).

        Parameters
        ----------
        timeseries : pd.DataFrame
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        return (
            timeseries.groupby(pd.PeriodIndex(timeseries[date_column], freq=time_period))[target_column]
            .mean()
            .reset_index(drop=False)
        )

    def integrety_check_timeseries(self: Any, time_series_1: pd.DataFrame, time_series_2: pd.DataFrame) -> bool:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        time_series_1 : pd.DataFrame
            _description_
        time_series_2 : pd.DataFrame
            _description_

        Returns
        -------
        bool
            _description_
        """
        if time_series_1.shape[0] != time_series_2:
            return False
        else:
            return True

    def correlation_coefficient(self: Any, time_series_1: pd.Series, time_series_2: pd.Series) -> Tuple[float, float]:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        time_series_1 : pd.Series
            _description_
        time_series_2 : pd.Series
            _description_

        Returns
        -------
        Tuple[float, float]
            _description_
        """
        corr = pearsonr(time_series_1, time_series_2)

        return corr.statistic, corr.pvalue

    def crosscorrelation(
        self: Any, time_series_1: pd.Series, time_series_2: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        time_series_1 : pd.Series
            _description_
        time_series_2 : pd.Series
            _description_

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            _description_
        """
        time_series_1 = (time_series_1 - time_series_1.mean()) / (time_series_1.std() * time_series_1.shape[0])
        time_series_2 = (time_series_2 - time_series_2.mean()) / (time_series_2.std())

        return correlate(time_series_1, time_series_2), correlation_lags(time_series_1.shape[0], time_series_2.shape[0])

    def max_crosscorrelation(self: Any, time_series_1: pd.Series, time_series_2: pd.Series) -> Tuple[float, int]:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        time_series_1 : pd.Series
            _description_
        time_series_2 : pd.Series
            _description_

        Returns
        -------
        Tuple[float, int]
            _description_
        """
        corr, corr_lags = self.crosscorrelation(time_series_1=time_series_1, time_series_2=time_series_2)

        return corr_lags[np.argmax(np.abs(corr))], np.argmax(np.abs(corr))

    def temporal_offset(
        self: Any, time_series_1: pd.Series, time_series_2: pd.Series, metric: str = "mean"
    ) -> Tuple[float, Union[int, None]]:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        time_series_1 : pd.Series
            _description_
        time_series_2 : pd.Series
            _description_
        metric : str, optional
            _description_, by default "mean"

        Returns
        -------
        Tuple[float, Union[int, None]]
            _description_
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
        self: Any, time_series_1: pd.Series, time_series_2: pd.Series, plot: bool=False
    ) -> Tuple[float, float, float, float, float]:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        time_series_1 : pd.Series
            _description_
        time_series_2 : pd.Series
            _description_
        plot : bool, optional
            _description_, by default False

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
        _summary_.

        Parameters
        ----------
        time_series_1 : pd.Series
            _description_
        time_series_2 : pd.Series
            _description_
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
