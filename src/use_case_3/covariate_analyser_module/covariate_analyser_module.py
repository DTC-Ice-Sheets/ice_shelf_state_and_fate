"""Module for undertaking the covariate analysis."""

from __future__ import annotations

from copy import deepcopy
from textwrap import wrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RANSACRegressor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.tsatools import lagmat2ds

from use_case_3.covariate_analyser_module.timeseries_preprocessor import (
    TimeSeriesPreprocessor,
)
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    JointPlotAnalysisType,
    ModelEvaluationMetric,
)

SUBGLACIAL_LAKE_LABELS = ["volume_change", "lake_discharge"]
BASAL_MELT_RATE_LABELS = ["land_ice_basal_melt_rate", "basal_melt_anomalies"]


class CovariateAnalyser:
    """The covariate analysis module.

    TODO: None of the plotting functions are tested!

    NOTE: It is generally considered unsafe to use the timeseries in the self._timeseries_altered attribute as
    these have been altered with no history of how and when. There are no gaurentees that a timeseries label which
    is in self.timeseries is in self._timeseries_altered. Only use these timeseries if you know what is stored
    there!

    Parameters
    ----------
    timeseries_container : TimeSeriesPreprocessor
        The timeseries container.
    alpha : float
        The default significance value used in any hypothesis tests, by default 0.05
    """

    def __init__(
        self: CovariateAnalyser,
        timeseries_container: TimeSeriesPreprocessor,
        alpha: float = 0.05,
    ) -> None:
        self.alpha = alpha  # The critical value
        self.colours = [  # Some colours for the plots to use.
            "black",
            "coral",
            "forestgreen",
            "mediumslateblue",
            "lightseagreen",
            "dodgerblue",
            "darkcyan",
            "gold",
            "lightsteelblue",
            "darkslategrey",
            "darkmagenta",
            "crimson",
        ]
        self.timeseries_container = timeseries_container
        self._timeseries_altered = {}  # Store any timeseries objects requiring temporary processing.

    def covariate_analysis(
        self: CovariateAnalyser,
        axis_labels: list | None = None,
        max_lag: int | None = None,
        analysis_type: JointPlotAnalysisType = JointPlotAnalysisType.CORR,
    ) -> tuple[plt.figure, plt.axes]:
        """
        Conduct the covariate analysis.

        This will be the main entrypoint to run the covariate analysis. Currently this
        only runs one method, but the final version will output multiple plots and statistics.

        Parameters
        ----------
        axis_labels : list | None
            A list of labels which can be used to customise the axis labels if required, by default None
        max_lag : int | None
            The maximum allowable lag allowed to align the time series. This can be forward or backwards in time, by
            default None
        analysis_type : JointPlotAnalysisType
            Whether or not the crosscorrelation is used to align the series, by default JointPlotAnalysisType.CORR

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The grid joint plots of the covariate analysis.
        """
        return self.jointplot_grid(axis_labels=axis_labels, max_lag=max_lag, analysis_type=analysis_type)

    def find_correlated_vars(
        self: CovariateAnalyser, corr_thresh: float = 0.5
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """
        Find timeseries which are correlated.

        This method finds correlated variables where the correlation is greater than a user defined threshold and is
        statistically significant. Note it returns statistcally significant correlations which are both positive and
        negative.

        Parameters
        ----------
        corr_thresh : float
            The correlation threshold, by default 0.5

        Returns
        -------
        dict[tuple[str, str], tuple[float, float]]
            A dictionary with the label pairs as the key with the associated correlation coefficent and p_value.
        """
        correlated_vars = {}
        for timeseries_label_1 in self.timeseries_container.timeseries:
            for timeseries_label_2 in self.timeseries_container.timeseries:
                if (
                    timeseries_label_1 != timeseries_label_2
                    and (timeseries_label_1, timeseries_label_2) not in correlated_vars
                    and (timeseries_label_2, timeseries_label_1) not in correlated_vars
                ):
                    corr_coef, p_value = self.correlation_coefficient(
                        timeseries_label_1, timeseries_label_2, altered=False
                    )
                    # We only store a correlated variable if it is not already in the dict.
                    if np.abs(corr_coef) >= corr_thresh and p_value <= self.alpha:
                        correlated_vars[timeseries_label_1, timeseries_label_2] = (corr_coef, p_value)

        return correlated_vars

    def find_crosscorrelated_vars(
        self: CovariateAnalyser,
        corr_thresh: float = 0.5,
        max_lag: int | None = 3,
    ) -> dict[tuple[str, str], tuple[float, float, np.int64 | int | float]]:
        """
        Find timeseries which are correlated with associated lag.

        This method finds correlated variables where the correlation is greater than a user defined threshold and is
        statistically significant. Note it returns statistcally significant correlations which are both positive and
        negative. The cross correlation is computed and correlation with the largest magnitude is used.

        Parameters
        ----------
        corr_thresh : float
            The correlation threshold, by default 0.5
        max_lag : int | None
            The maximum lag to consider, not this will apply this maximum lag in both the positive and the negative
            directions, by default 3

        Returns
        -------
        dict[tuple[str, str], tuple[float, float, np.int64 | int | float]]
            A dictionary with the label pairs as the key with the associated correlation coefficent, p_value and lag.
        """
        correlated_vars = {}
        for timeseries_label_1 in self.timeseries_container.timeseries:
            for timeseries_label_2 in self.timeseries_container.timeseries:
                if (
                    timeseries_label_1 != timeseries_label_2
                    and (timeseries_label_1, timeseries_label_2) not in correlated_vars
                    and (timeseries_label_2, timeseries_label_1) not in correlated_vars
                ):
                    # First we allign the timeseries in time using the cross-correlation.
                    lag, _ = self._align_timeseries(
                        timeseries_label_1,
                        timeseries_label_2,
                        max_lag=max_lag,
                    )
                    # We now calculate the corr_coef and p_value using the alligned time series
                    corr_coef, p_value = self.correlation_coefficient(
                        timeseries_label_1=timeseries_label_1,
                        timeseries_label_2=timeseries_label_2,
                        altered=True,
                    )
                    # We only store a correlated variable if it is not already in the dict.
                    if np.abs(corr_coef) >= corr_thresh and p_value <= self.alpha:
                        correlated_vars[timeseries_label_1, timeseries_label_2] = (corr_coef, p_value, lag)

        return correlated_vars

    def find_causal_vars(
        self: CovariateAnalyser,
        max_lags: int | None = 10,
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """
        Find variables which are causally related. This method finds causal variables using the p-value.

        Parameters
        ----------
        max_lags : int | None
            The maximum number of lags, if to large will be adjusted to floor((n - 1) / 3) - 1 where n is the no. rows
             in data. In this context the maximum lags refers to the number of lags used to forecast, by default 10

        Returns
        -------
        dict[tuple[str, str], tuple[float, float]]
            A dictionary with the label pairs as the key with the associated fscore, and p_value.
        """
        causal_vars = {}
        for timeseries_label_1 in self.timeseries_container.timeseries:
            for timeseries_label_2 in self.timeseries_container.timeseries:
                if (
                    timeseries_label_1 != timeseries_label_2
                    and (timeseries_label_1, timeseries_label_2) not in causal_vars
                ):
                    # Calculate the f_score and p_value using the Granger Causality test
                    f_score, p_value = self._granger_causality_test(
                        timeseries_label_1=timeseries_label_1,
                        timeseries_label_2=timeseries_label_2,
                        max_lags=max_lags,
                        metric=ModelEvaluationMetric.BIC,
                    )
                    # We only store a correlated variable if it is not already in the dict. Note causality is
                    # directional!
                    if p_value <= self.alpha:
                        causal_vars[timeseries_label_1, timeseries_label_2] = (f_score, p_value)

        return causal_vars

    def correlation_coefficient(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str, altered: bool = False
    ) -> tuple[float, float]:
        """
        Calculate the Pearson Correlation Coefficient between two time series.

        Parameters
        ----------
        timeseries_label_1 : str
            The first time series.
        timeseries_label_2 : str
            The second time series.
        altered : bool
            A boolean flag specifying whether the original timeseries or the temporarily altered timeseries are used.

        Returns
        -------
        tuple[float, float]
            The Pearson correlation coefficient and associated p-value.

        Raises
        ------
        ValueError
            "The labels specified are not in the altered timeseries container"
        """
        if altered:
            if timeseries_label_1 not in self._timeseries_altered or timeseries_label_2 not in self._timeseries_altered:
                raise ValueError("The labels specified are not in the altered timeseries container")
            timeseries_1 = self._timeseries_altered[timeseries_label_1]
            timeseries_2 = self._timeseries_altered[timeseries_label_2]
        else:
            timeseries_1 = self.timeseries_container.timeseries[timeseries_label_1]
            timeseries_2 = self.timeseries_container.timeseries[timeseries_label_2]

        corr = pearsonr(
            timeseries_1.measurements,
            timeseries_2.measurements,
        )

        return corr.statistic, corr.pvalue

    def covariance(
        self: CovariateAnalyser, timeseries_labels: list[str], normalisation: str | None = None, altered: bool = False
    ) -> pd.DataFrame:
        """
        Calculate the normalised covariance matrix between a set of timeseries.

        Parameters
        ----------
        timeseries_labels : list[str]
            The list of timeseries labels to include in the covariance matrix.
        normalisation : str | None
            The type of normalisation applied, this can be None, or corrcoef.
        altered : bool
            A boolean flag specifying whether the original timeseries or the temporarily altered timeseries are used.

        Returns
        -------
        pd.DataFrame
            The covariance matrix.
        """
        if altered:
            timeseries = [ts for ts in self._timeseries_altered.values() if ts.label in timeseries_labels]
        else:
            timeseries = [ts for ts in self.timeseries_container.timeseries.values() if ts.label in timeseries_labels]

        if normalisation == "mean_centered":
            timeseries = [ts.measurements - ts.measurements.mean() for ts in timeseries]
        else:
            timeseries = [ts.measurements for ts in timeseries]

        df = pd.concat(timeseries, axis=1)
        if normalisation == "corrcoef":
            return pd.DataFrame(np.corrcoef(df.T), columns=df.columns, index=df.columns)
        return pd.DataFrame(np.cov(df.T), columns=df.columns, index=df.columns)

    def crosscorrelation(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str, altered: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the cross-correlation between two timeseries.

        Parameters
        ----------
        timeseries_label_1 : str
            The first time series.
        timeseries_label_2 : str
            The second time series.
        altered : bool
            A boolean flag specifying whether the original timeseries or the temporarily altered timeseries are used.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The cross-correlation with associated lags.
        """
        if altered:
            timeseries_1 = deepcopy(self._timeseries_altered[timeseries_label_1])
            timeseries_2 = deepcopy(self._timeseries_altered[timeseries_label_2])
        else:
            timeseries_1 = deepcopy(self.timeseries_container.timeseries[timeseries_label_1])
            timeseries_2 = deepcopy(self.timeseries_container.timeseries[timeseries_label_2])
        timeseries_1 = (timeseries_1.measurements - timeseries_1.measurements.mean()) / np.std(
            timeseries_1.measurements
        )
        timeseries_2 = (timeseries_2.measurements - timeseries_2.measurements.mean()) / np.std(
            timeseries_2.measurements
        )

        return correlate(timeseries_1, timeseries_2, "full") / min(
            timeseries_1.shape[0], timeseries_2.shape[0]
        ), correlation_lags(timeseries_1.shape[0], timeseries_2.shape[0])

    def max_crosscorrelation(
        self: CovariateAnalyser,
        timeseries_label_1: str,
        timeseries_label_2: str,
        max_lag: int | np.int64 | None = None,
        altered: bool = False,
    ) -> tuple[np.float64, np.int64]:
        """
        Find the maximum cross-correlation and associated lag.

        Parameters
        ----------
        timeseries_label_1 : str
            The label for the first timeseries.
        timeseries_label_2 : str
            The label for the second timeseries.
        max_lag : int | np.int64 | None
            The maximum lag to consider, not this will apply this maximum lag in both the positive and the negative
            directions. If set to None there is no maximum lag, by default None
        altered : bool
            A boolean flag specifying whether the original timeseries or the temporarily altered timeseries are used.

        Returns
        -------
        tuple[np.float64, np.int64]
            The maximum cross-correlation and associated lag.
        """
        corr, corr_lags = self.crosscorrelation(
            timeseries_label_1=timeseries_label_1,
            timeseries_label_2=timeseries_label_2,
            altered=altered,
        )

        if max_lag is None:
            max_lag = max(corr_lags)

        corr = [corr[idx] for idx in range(len(corr_lags)) if np.abs(corr_lags[idx]) <= max_lag]
        corr_lags = [corr_lags[idx] for idx in range(len(corr_lags)) if np.abs(corr_lags[idx]) <= max_lag]

        return corr[np.argmax(np.abs(corr))], corr_lags[np.argmax(np.abs(corr))]

    def _align_timeseries(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str, max_lag: int | None = None
    ) -> tuple[int | np.int64, AggregationPeriod]:
        """
        Align two timeseries according to the maximum cross correlation.

        Parameters
        ----------
        timeseries_label_1 : str
            The label for the first timeseries.
        timeseries_label_2 : str
            The label for the second timeseries.
        max_lag : int | None
            The maximum lag to consider, note this will apply this maximum lag in both the positive and the negative
            directions. If set to None there is no maximum lag, by default None

        Returns
        -------
        tuple[int | np.int64, AggregationPeriod]
            The lag as an integer, and its units as an AggregationPeriod enum string.
        """
        _, corr_lag = self.max_crosscorrelation(
            timeseries_label_1=timeseries_label_1,
            timeseries_label_2=timeseries_label_2,
            max_lag=max_lag,
        )

        timeseries_1 = deepcopy(self.timeseries_container.timeseries[timeseries_label_1])
        timeseries_2 = deepcopy(self.timeseries_container.timeseries[timeseries_label_2])

        freq = self.timeseries_container.aggregation_period

        offset_map = {
            AggregationPeriod.MONTH: pd.offsets.MonthBegin,
            AggregationPeriod.QUARTER: pd.offsets.QuarterBegin,
            AggregationPeriod.YEAR: pd.offsets.YearBegin,
            AggregationPeriod.DAY: pd.offsets.Day,
        }

        offset_func = offset_map[freq]
        offset = (
            pd.offsets.QuarterBegin(corr_lag, startingMonth=1)
            if freq == AggregationPeriod.QUARTER
            else offset_func(corr_lag)
        )

        timeseries_2.measurements.index = timeseries_2.measurements.index + offset
        if timeseries_2.measurement_uncertainties is not None:
            timeseries_2.measurement_uncertainties.index = timeseries_2.measurement_uncertainties.index + offset

        # Find the latest start datetime for all the input timeseries
        start_date = np.max([np.min(timeseries_1.measurements.index), np.min(timeseries_2.measurements.index)])
        # Find the earliest end datetime for all the input timeseries
        end_date = np.min([np.max(timeseries_1.measurements.index), np.max(timeseries_2.measurements.index)])

        for ts in [timeseries_1, timeseries_2]:
            ts.measurements = ts.measurements[
                (ts.measurements.index >= start_date) & (ts.measurements.index <= end_date)
            ]
            if ts.measurement_uncertainties is not None:
                ts.measurement_uncertainties = ts.measurement_uncertainties[
                    (ts.measurement_uncertainties.index >= start_date)
                    & (ts.measurement_uncertainties.index <= end_date)
                ]

            self._timeseries_altered[ts.label] = ts

        return (
            corr_lag,
            freq,
        )

    def temporal_offset(
        self: CovariateAnalyser,
        timeseries_label_1: str,
        timeseries_label_2: str,
        metric: AggregationStrategy = AggregationStrategy.MEAN,
    ) -> tuple[float | int | np.float64 | np.int64, np.ndarray | int | np.int64 | None] | None:
        """
        Calculate a temporal offset.

        This is the difference between two series. This can be done using various
        statistics such as the mean, median, maximum and minimum.

        Parameters
        ----------
        timeseries_label_1 : str
            The label for the first timeseries.
        timeseries_label_2 : str
            The label for the second timeseries.
        metric : AggregationStrategy
            The statistic to quantify the difference between the two time series, by default "mean".

        Returns
        -------
        tuple[float | int | np.float64 | np.int64, np.ndarray | int | np.int64 | None] | None
            The temporal offset and if applicable the associated index.
        """
        diff = (
            self.timeseries_container.timeseries[timeseries_label_1].measurements
            - self.timeseries_container.timeseries[timeseries_label_2].measurements
        )

        if metric == AggregationStrategy.MEAN:
            return diff.mean(), None
        if metric == AggregationStrategy.MEDIAN:
            return diff.median(), None
        if metric == AggregationStrategy.MAX:
            return diff.max(), diff.argmax()
        if metric == AggregationStrategy.MIN:
            return diff.min(), diff.argmin()

        return None

    def temporal_slope(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str, plot: bool = False
    ) -> tuple[float, float, float, float, float]:
        """
        Calculate the temporal slope of two time series using robust linear regression (RANSac).

        Parameters
        ----------
        timeseries_label_1 : str
            The label for the first timeseries.
        timeseries_label_2 : str
            The label for the second timeseries.
        plot : bool
            True if you want a plot of the model fits, by default False

        Returns
        -------
        tuple[float, float, float, float, float]
            A tuple containing the ratio of gradients between timeseries 1 and timeseries 2, the slope of timeseries 1,
            the intercept of timeseries 1, the slope of timeseries 2, and the intercept of timeseries 2.
        """
        reg_1 = RANSACRegressor(random_state=0).fit(
            np.arange(0, self.timeseries_container.timeseries[timeseries_label_1].measurements.shape[0]).reshape(-1, 1),
            self.timeseries_container.timeseries[timeseries_label_1].measurements,
        )
        reg_2 = RANSACRegressor(random_state=0).fit(
            np.arange(0, self.timeseries_container.timeseries[timeseries_label_2].measurements.shape[0]).reshape(-1, 1),
            self.timeseries_container.timeseries[timeseries_label_2].measurements,
        )

        if plot:
            _fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].plot(
                np.arange(0, self.timeseries_container.timeseries[timeseries_label_1].measurements.shape[0]).reshape(
                    -1, 1
                ),
                self.timeseries_container.timeseries[timeseries_label_1].measurements,
                ".k",
            )
            axes[0].plot(
                [0, self.timeseries_container.timeseries[timeseries_label_1].measurements.shape[0]],
                [
                    reg_1.estimator_.intercept_,
                    (
                        reg_1.estimator_.coef_[0]
                        * self.timeseries_container.timeseries[timeseries_label_1].measurements.shape[0]
                    )
                    + reg_1.estimator_.intercept_,
                ],
                "--k",
            )
            axes[0].grid()
            axes[1].plot(
                np.arange(0, self.timeseries_container.timeseries[timeseries_label_2].measurements.shape[0]).reshape(
                    -1, 1
                ),
                self.timeseries_container.timeseries[timeseries_label_2].measurements,
                ".k",
            )
            axes[1].plot(
                [0, self.timeseries_container.timeseries[timeseries_label_2].measurements.shape[0]],
                [
                    reg_2.estimator_.intercept_,
                    (
                        reg_2.estimator_.coef_[0]
                        * self.timeseries_container.timeseries[timeseries_label_2].measurements.shape[0]
                    )
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

    @staticmethod
    def bin_data(x: pd.Series, y: pd.Series, n_bins: int, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin paired observations and calculate the mean values in each bin.

        Drop rows with missing values before binning. Use equal-width bins
        by default, or equal-frequency bins when ``method`` is ``"quantile"``.

        Parameters
        ----------
        x : pd.Series
            Provide the values used to determine bin membership.
        y : pd.Series
            Provide the paired values to average within each bin.
        n_bins : int
            Set the number of bins to create.
        method : str
            Choose ``"width"`` for equal-width bins or ``"quantile"`` for
            equal-frequency bins.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Return the mean x values, mean y values, and observation counts
            for each non-empty bin.

        Raises
        ------
        TypeError
            Raise when either input is not a pandas Series.
        ValueError
            Raise if method must be 'width' or 'quantile'
        """
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise TypeError("x and y must be pandas Series")
        df = pd.DataFrame({"x": x, "y": y}) if x.index.equals(y.index) else pd.concat([x, y], axis=1, keys=["x", "y"])
        df = df.astype(float).dropna()
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        if method == "width":
            edges = np.linspace(x.min(), x.max(), n_bins + 1)
        elif method == "quantile":
            edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        else:
            raise ValueError("method must be 'width' or 'quantile'")

        idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)

        xb, yb, nb = [], [], []
        for k in range(n_bins):
            s = idx == k
            if s.any():
                xb.append(x[s].mean())
                yb.append(y[s].mean())
                nb.append(int(s.sum()))
        return np.array(xb), np.array(yb), np.array(nb)

    @staticmethod
    def _power_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Evaluate the power-law model for the given input values.

        Parameters
        ----------
        x : np.ndarray
            Provide the independent variable values.
        a : float
            Set the multiplicative coefficient.
        b : float
            Set the exponent.

        Returns
        -------
        np.ndarray
            Return values calculated as ``a * x**b``.
        """
        return a * x**b

    @staticmethod
    def fit_power_law_nonlinear(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
        """
        Fit a nonlinear power-law model and estimate confidence intervals.

        Fit the model ``y = a * x**b`` using several initial estimates for
        the exponent, then select the fit with the lowest residual sum of
        squares.

        Parameters
        ----------
        x : np.ndarray
            Provide the independent variable values.
        y : np.ndarray
            Provide the dependent variable values.
        alpha : float
            Set the significance level used to calculate confidence
            intervals for the model parameters.

        Returns
        -------
        dict
            Return the fitted parameters ``a`` and ``b``, their confidence
            intervals, residual sum of squares (``rss``), coefficient of
            determination (``r2``), and sample count (``n``).

        Raises
        ------
        ValueError
            Raise when any ``x`` value is negative or there are not more
            than two data points.
        RuntimeError
            Raise when the fit fails for every initial exponent estimate.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if np.any(x < 0):
            raise ValueError("x must be >= 0 for a power law")
        n_par = 2
        if len(x) <= n_par:
            raise ValueError(f"need more than {n_par} points")

        scale = np.ptp(y) / max(np.ptp(x), 1e-12)
        best = None
        for b0 in (0.25, 0.5, 1.0, 1.5, 2.0):
            p0 = [scale, b0]
            try:
                p, cov = curve_fit(CovariateAnalyser._power_model, x, y, p0=p0, maxfev=50000)
            except (RuntimeError, ValueError):
                continue
            rss = float(np.sum((y - CovariateAnalyser._power_model(x, *p)) ** 2))
            if best is None or rss < best[0]:
                best = (rss, p, cov)
        if best is None:
            raise RuntimeError("fit did not converge from any starting point")

        rss, p, cov = best
        se = np.sqrt(np.diag(cov))
        t = stats.t.ppf(1 - alpha / 2, len(x) - n_par)
        tss = float(np.sum((y - y.mean()) ** 2))
        return {
            "a": float(p[0]),
            "b": float(p[1]),
            "a_ci": (float(p[0] - t * se[0]), float(p[0] + t * se[0])),
            "b_ci": (float(p[1] - t * se[1]), float(p[1] + t * se[1])),
            "rss": rss,
            "r2": float(1 - rss / tss) if tss > 0 else float("nan"),
            "n": len(x),
        }

    @staticmethod
    def detrend(s: pd.Series) -> pd.Series:
        """
        Remove a robust linear trend from a time series.

        Sort the series by its datetime index, fit a RANSAC linear model
        against elapsed time in years, and subtract the fitted trend.

        Parameters
        ----------
        s : pd.Series
            Provide the time series to detrend.

        Returns
        -------
        pd.Series
            Return the detrended values with a sorted datetime index and
            the input series name.
        """
        s = s.sort_index()

        t0 = s.index[0]
        x = ((s.index - t0) / pd.Timedelta(days=365.25)).to_numpy().reshape(-1, 1)
        y = s.to_numpy()

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=None,
            random_state=0,
        )
        ransac.fit(x, y)

        trend = pd.Series(ransac.predict(x), index=s.index, name="trend")
        return (s - trend).rename(s.name)

    def fit_power_law_model(
        self: CovariateAnalyser,
        timeseries_label_1: str,
        timeseries_label_2: str,
        max_lag: int = 6,
        n_bins: int = 8,
        method: str = "width",
        x_anomalies: bool = False,
        y_anomalies: bool = False,
    ) -> dict:
        """
        Align two time series and fit a power-law model to their binned data.

        Remove bins with non-positive mean x values before fitting.

        Parameters
        ----------
        timeseries_label_1 : str
            Select the time series used as the independent variable.
        timeseries_label_2 : str
            Select the time series used as the dependent variable.
        max_lag : int
            Limit the lag considered when aligning the time series.
        n_bins : int
            Set the number of bins used to summarize paired observations.
        method : str
            Choose the binning method passed to `bin_data`.
        x_anomalies : bool
            Indicate whether to detrend the independent variable time series.
        y_anomalies : bool
            Indicate whether to detrend the dependent variable time series.

        Returns
        -------
        dict
            Return the fitted model statistics, including the lag used to align
            the time series.
        """
        corr_lag, _ = self._align_timeseries(timeseries_label_1, timeseries_label_2, max_lag=max_lag)

        x = (
            self._timeseries_altered[timeseries_label_1].measurements
            if not x_anomalies
            else self.detrend(self._timeseries_altered[timeseries_label_1].measurements)
        )

        # We invert the sign as we are interested in the rate of lake discharge which is the inverse of volume change.
        if "volume_change" in self._timeseries_altered[timeseries_label_1].measurement_name:
            x = -1 * x

        y = (
            self._timeseries_altered[timeseries_label_2].measurements
            if not y_anomalies
            else self.detrend(self._timeseries_altered[timeseries_label_2].measurements)
        )

        x_bin, y_bin, _ = self.bin_data(x=x, y=y, n_bins=n_bins, method=method)

        filt_mask = x_bin > 0
        x_bin, y_bin = x_bin[filt_mask], y_bin[filt_mask]

        model_results = self.fit_power_law_nonlinear(x_bin, y_bin)
        model_results["x_bin"] = x_bin
        model_results["y_bin"] = y_bin
        model_results["corr_lag"] = corr_lag
        model_results["x"] = x.to_numpy()
        model_results["y"] = y.to_numpy()
        model_results["datetime"] = x.index.to_numpy()
        model_results["timeseries_label_1"] = timeseries_label_1
        model_results["timeseries_label_2"] = timeseries_label_2

        return model_results

    @staticmethod
    def predict_power_law(x: np.ndarray, model: dict) -> np.ndarray:
        """
        Predict values from a fitted power-law model.

        Parameters
        ----------
        x : np.ndarray
            Provide the independent variable values at which to predict.
        model : dict
            Provide fitted model parameters with ``a`` and ``b`` keys.

        Returns
        -------
        np.ndarray
            Return predictions calculated as ``a * x**b``.

        Raises
        ------
        ValueError
            Raise when any input ``x`` value is negative.
        """
        arr = np.asarray(x, float)
        if np.any(arr < 0):
            raise ValueError("x must be >= 0 for a power law")
        return model["a"] * arr ** model["b"]

    def jointplot(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str, linear: bool = True
    ) -> plt.plot:
        """
        Visualise a joint plot between two time series. This returns a plot.

        TODO: Currently not necessarily used, but will lieky be useful so leaving currently.

        Parameters
        ----------
        timeseries_label_1 : str
            The label for the first timeseries.
        timeseries_label_2 : str
            The label for the second timeseries.
        linear : bool
            If true a linear fit is displayed using robust linear regression, by default True

        Returns
        -------
        plt.plot
            The plot object.
        """
        reg_1 = LinearRegression().fit(
            self.timeseries.get_timeseries([timeseries_label_1])[0].to_numpy().reshape(-1, 1),
            self.timeseries.get_timeseries([timeseries_label_2])[0].to_numpy().reshape(-1, 1),
        )

        joint = sns.jointplot(
            x=self.timeseries.get_timeseries([timeseries_label_1])[0],
            y=self.timeseries.get_timeseries([timeseries_label_2])[0],
        )
        if linear:
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

        return joint

    def run_analysis(  # noqa: C901
        self: CovariateAnalyser,
        max_lag: int | None = 6,
        analysis_type: JointPlotAnalysisType = JointPlotAnalysisType.CORR,
    ) -> dict:
        """
        Perform an analysis of the covariate relationships between the timeseries.

        Parameters
        ----------
        max_lag : int | None
            The maximum lag to consider for the analysis, by default 6 time units.
        analysis_type : JointPlotAnalysisType
            The type of analysis to perform, by default JointPlotAnalysisType.CORR.

        Returns
        -------
        dict
            A dictionary containing the results of the analysis.
        """
        out = {
            "variables": {},
            "covariate_stats": {},
        }

        for ts in self.timeseries_container.timeseries.values():
            hist, bin_edges = np.histogram(ts.measurements[~ts.measurements.isna()])
            ts.summary_statistics()
            out["variables"][ts.label] = {
                "measurement_name": ts.measurement_name,
                "values": ts.measurements.to_list(),
                "values_uncertainties": ts.measurement_uncertainties.to_list()
                if ts.measurement_uncertainties is not None
                else [],
                "summary_statistics": ts.summary_stats,
                "timestamps": ts.measurements.index.strftime("%Y-%m-%d").to_list(),
                "hist_values": hist.tolist(),
                "hist_edges": bin_edges.tolist(),
            }

            try:
                change_points, _ = ts.detect_changes(jump=1, penalty=25000.0, min_size=1)
                out["variables"][ts.label]["change_points"] = change_points.to_list()
            except Exception:  # noqa: BLE001
                out["variables"][ts.label]["change_points"] = []
            try:
                anomolies, anomalies_upper, anomalies_lower = ts.detect_anomalies()
                out["variables"][ts.label]["anomolies"] = anomolies.to_list()
                out["variables"][ts.label]["anomalies_upper"] = anomalies_upper.to_list()
                out["variables"][ts.label]["anomalies_lower"] = anomalies_lower.to_list()

            except Exception:  # noqa: BLE001
                out["variables"][ts.label]["anomolies"] = []
                out["variables"][ts.label]["anomalies_upper"] = []
                out["variables"][ts.label]["anomalies_lower"] = []

        for ts_a in self.timeseries_container.timeseries.values():
            for ts_b in self.timeseries_container.timeseries.values():
                if ts_a.label == ts_b.label:
                    continue
                if (
                    f"{ts_b.label}/{ts_a.label}" in out["covariate_stats"]
                    and analysis_type != JointPlotAnalysisType.CAUSAL
                ):
                    continue
                key = f"{ts_a.label}/{ts_b.label}"

                stats = {}
                if analysis_type == JointPlotAnalysisType.CROSS_CORR:
                    lag, lag_units = self._align_timeseries(
                        timeseries_label_1=ts_a.label, timeseries_label_2=ts_b.label, max_lag=max_lag
                    )
                    corr, p_value = self.correlation_coefficient(
                        timeseries_label_1=ts_a.label,
                        timeseries_label_2=ts_b.label,
                        altered=True,
                    )
                    stats |= {
                        "Aligned Lag": AggregationPeriod.format_with_units(lag, lag_units),
                        "Correlation Coefficient": corr,
                        "P-Value": p_value,
                    }
                elif analysis_type == JointPlotAnalysisType.CORR:
                    corr, p_value = self.correlation_coefficient(
                        timeseries_label_1=ts_a.label,
                        timeseries_label_2=ts_b.label,
                    )
                    stats |= {
                        "Correlation Coefficient": corr,
                        "P-Value": p_value,
                    }
                elif analysis_type == JointPlotAnalysisType.CAUSAL:
                    f_score, p_value = self._granger_causality_test(
                        timeseries_label_1=ts_a.label,
                        timeseries_label_2=ts_b.label,
                        max_lags=max_lag or 20,
                        metric=ModelEvaluationMetric.BIC,
                    )
                    stats |= {
                        "F-Score": f_score,
                        "P-Value": p_value,
                    }
                out["covariate_stats"][key] = stats

                # Conduct the power analysis, we only do this for specific combinations of variables. I think in the
                # future this should be added as part of the variables yaml, but currently this is fine to hardcode.
                # label 1 --> Related to subglacial lake discharge
                # label 2 --> Related to basal melt rate
                try:
                    x_flag, label_x = self._determine_if_label_in_power_law(ts_a, ts_b, "x")
                    y_flag, label_y = self._determine_if_label_in_power_law(ts_a, ts_b, "y")

                    if x_flag and y_flag:
                        power_law_results = self.fit_power_law_model(
                            timeseries_label_1=label_x,
                            timeseries_label_2=label_y,
                        )
                        power_law_results = {
                            key: self._safe_serialise(value) for key, value in power_law_results.items()
                        }

                        out.setdefault("power_law_analysis", {})[key] = power_law_results
                except Exception as e:  # noqa: BLE001
                    out.setdefault("power_law_analysis", {})[key] = {"error": str(e)}

        return out

    @staticmethod
    def _safe_serialise(x: Any) -> Any:  # noqa: ANN401
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.datetime64):
                return x.astype("datetime64[s]").astype(str).tolist()
            return x.tolist()
        if isinstance(x, np.datetime64):
            return str(x)
        if isinstance(x, np.generic):
            return x.item()
        return x

    def _determine_if_label_in_power_law(self, ts_a: Timeseries, ts_b: Timeseries, axis: str) -> tuple[bool, str]:
        """Check whether a timeseries label is eligible for power-law analysis.

        The x axis accepts labels containing a subglacial-lake variable name;
        the y axis accepts labels containing a basal-melt-rate variable name.

        Parameters
        ----------
        ts_a : Timeseries
            First timeseries to check.
        ts_b : Timeseries
            Second timeseries to check.
        axis : str
            Power-law axis to check: ``"x"`` for subglacial-lake labels or
            ``"y"`` for basal-melt-rate labels.

        Returns
        -------
        tuple[bool, str]
            A tuple where the first element indicates whether the label contains a variable name associated with the
            specified axis, and the second element is the label that matches the axis criteria (or an empty string if
            none match).
        """
        if axis == "x":
            variable_labels = SUBGLACIAL_LAKE_LABELS
        elif axis == "y":
            variable_labels = BASAL_MELT_RATE_LABELS
        else:
            return False, ""

        for candidate in (ts_a, ts_b):
            if any(variable in candidate.measurement_name for variable in variable_labels):
                return True, candidate.label
        return False, ""

    def plot_timeseries(self: CovariateAnalyser, yaxis_labels: list | None = None) -> tuple[plt.figure, plt.axes]:
        """
        Plot all the timeseries in the timeseries container.

        Parameters
        ----------
        yaxis_labels : list | None
            A list of labels which can be used to customise the y-axis labels if required, by default None.

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figure and axes.
        """
        timeseries_labels = list(self.timeseries_container.timeseries)
        f, axes = plt.subplots(
            len(timeseries_labels),
            1,
            figsize=(10, 4 * len(timeseries_labels)),
            sharex="col",
            squeeze=False,
        )
        axes = axes.flatten()

        for n, _ in enumerate(timeseries_labels):
            sns.lineplot(
                x=self.timeseries_container.timeseries[timeseries_labels[n]].measurements.index,
                y=self.timeseries_container.timeseries[timeseries_labels[n]].measurements.values,
                ax=axes[n],
                c=self.colours[n],
            )
            if self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties is not None:
                sns.lineplot(
                    x=self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.index,
                    y=self.timeseries_container.timeseries[timeseries_labels[n]].measurements.to_numpy()
                    + self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.to_numpy(),
                    ax=axes[n],
                    c=self.colours[n],
                    linestyle="--",
                    alpha=0.75,
                )
                sns.lineplot(
                    x=self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.index,
                    y=self.timeseries_container.timeseries[timeseries_labels[n]].measurements.to_numpy()
                    - self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.to_numpy(),
                    ax=axes[n],
                    c=self.colours[n],
                    linestyle="--",
                    alpha=0.75,
                )
                axes[n].fill_between(
                    self.timeseries_container.timeseries[timeseries_labels[n]].measurements.index,
                    self.timeseries_container.timeseries[timeseries_labels[n]].measurements.to_numpy()
                    + self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.to_numpy(),
                    self.timeseries_container.timeseries[timeseries_labels[n]].measurements.to_numpy()
                    - self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.to_numpy(),
                    alpha=0.5,
                    facecolor=self.colours[n],
                )
            yaxis_label = timeseries_labels[n] if yaxis_labels is None else yaxis_labels[n]
            axes[n].set_ylabel(
                "\n".join(wrap(yaxis_label, 15))
                + " \n ("
                + self.timeseries_container.timeseries[timeseries_labels[n]].unit
                + ")",
                fontsize=20,
            )
            axes[n].grid(True)
            axes[n].tick_params(axis="both", which="major", labelsize=15)
            axes[n].tick_params(axis="both", which="minor", labelsize=8)

        axes[n].set_xlabel("Year", fontsize=20)

        return f, axes

    @staticmethod
    def _granger_causality_test_bestlag(
        data: pd.DataFrame, max_lags: int, metric: ModelEvaluationMetric = ModelEvaluationMetric.BIC
    ) -> tuple[int, float]:
        """
        Fit OLS models iteratively to determine the number of lags which gives the lowest BIC or AIC score.

        Parameters
        ----------
        data : pd.DataFrame
            A 2d dataset with two variable columns.
        max_lags : int
            The maximum number of lags which must be <= floor((n - 1) / 3) - 1 where n is the no. rows in data.
        metric : ModelEvaluationMetric
            The evaluation metric used to evaluate the best model fit, by default ModelEvaluationMetric.BIC

        Returns
        -------
        tuple[int, float]
            The best lag and the lowest evaluation score.

        Raises
        ------
        ValueError
            "The metric must be one of ModelEvaluationMetric.BIC or ModelEvaluationMetric.AIC"
        """
        # Set some arbritrary values for the best model fits
        best_lag = None
        best_bic = np.inf

        # Loop through each lag and assess the model
        for lag in range(1, max_lags + 1):
            # Conver the data into an input matrix with lagged X and Y variables.
            dta = lagmat2ds(data, lag, trim="both", dropex=1)

            # We always add an intercept term to the model here
            dtajoint = add_constant(dta[:, 1:], prepend=False)
            model = OLS(dta[:, 0], dtajoint).fit()
            if metric == ModelEvaluationMetric.BIC:
                score = model.bic
            elif metric == ModelEvaluationMetric.AIC:
                score = model.aic
            else:
                raise ValueError("The metric must be one of ModelEvaluationMetric.BIC or ModelEvaluationMetric.AIC")
            # If the model performs better store the lag. Note this will prefer models with smaller lags if performance
            # is equal.
            if score < best_bic:
                best_lag = lag
                best_bic = model.bic

        return best_lag, best_bic

    def _granger_causality_test(
        self: CovariateAnalyser,
        timeseries_label_1: str,
        timeseries_label_2: str,
        max_lags: int,
        metric: ModelEvaluationMetric = ModelEvaluationMetric.BIC,
    ) -> tuple[float, float]:
        """
        Conduct a granger causality test to determine causality between two variables.

        This finds the lag (i.e. the number of lag variables in time for both X and Y) which results in the best
        model fit. Note, that in this case we are checking whether time_series_label_1 can be used to forecast
        time_series_2.

        Parameters
        ----------
        timeseries_label_1 : str
            The label of the first time series (X)
        timeseries_label_2 : str
            The variable for the second time series.
        max_lags : int
            The maximum number of lags, if to large will be adjusted to floor((n - 1) / 3) - 1 where n is the no. rows
             in data.
        metric : ModelEvaluationMetric
            The evaluation metric used to evaluate the best model fit, by default ModelEvaluationMetric.BIC

        Returns
        -------
        tuple[float, float]
            f-score and associated p-value.

        Raises
        ------
        ValueError
            "The number of lags must be greater than 0"
        """
        timeseries_1 = self.timeseries_container.timeseries[timeseries_label_1].measurements
        timeseries_2 = self.timeseries_container.timeseries[timeseries_label_2].measurements

        data = pd.concat([timeseries_1, timeseries_2], axis=1)
        max_lags_total = int((data.shape[0] - 1) / 3) - 1

        if max_lags_total < 1:
            raise ValueError("Data length is too short to perform Granger Causality test")

        if max_lags > max_lags_total:
            max_lags = max_lags_total

        if max_lags < 1:
            raise ValueError("The number of lags must be greater than 0")

        # Iteratively fit the data and determine the model with the lowest error
        best_lag, _ = self._granger_causality_test_bestlag(data=data, max_lags=max_lags, metric=metric)

        gc_res = grangercausalitytests(data, [best_lag], verbose=False)

        return gc_res[best_lag][0]["ssr_ftest"][0], gc_res[best_lag][0]["ssr_ftest"][1]
