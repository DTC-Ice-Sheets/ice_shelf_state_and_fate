"""Module for undertaking the covariate analysis."""

from __future__ import annotations

from copy import deepcopy
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    JointPlotAnalysisType,
    ModelEvaluationMetric,
)


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
        analysis_type : JointPlotAnalysisType, optional
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
        corr_thresh : float, optional
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
        corr_thresh : float, optional
            The correlation threshold, by default 0.5
        max_lag : int | None, optional
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
        max_lags : int | None, optional
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
        altered : bool, optional
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
            The maximum lag to consider, not this will apply this maximum lag in both the positive and the negative
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
            AggregationPeriod.MONTH: pd.offsets.MonthEnd,
            AggregationPeriod.QUARTER: pd.offsets.QuarterEnd,
            AggregationPeriod.YEAR: pd.offsets.YearEnd,
            AggregationPeriod.DAY: pd.offsets.Day,
        }

        offset_func = offset_map[freq]

        timeseries_2.measurements.index = timeseries_2.measurements.index + offset_func(corr_lag)
        if timeseries_2.measurement_uncertainties is not None:
            timeseries_2.measurement_uncertainties.index = timeseries_2.measurement_uncertainties.index + offset_func(
                corr_lag
            )

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
        plot : bool, optional
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

    def run_analysis(
        self: CovariateAnalyser,
        max_lag: int | None = None,
        analysis_type: JointPlotAnalysisType = JointPlotAnalysisType.CORR,
    ) -> dict:
        """
        Perform an analysis of the covariate relationships between the timeseries.

        Parameters
        ----------
        max_lag : int | None, optional
            The maximum lag to consider for the analysis, by default None.
        analysis_type : JointPlotAnalysisType, optional
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
            hist, bin_edges = np.histogram(ts.measurements)
            out["variables"][ts.label] = {
                "measurement_name": ts.measurement_name,
                "values": ts.measurements.to_list(),
                "values_uncertainties": ts.measurement_uncertainties.to_list()
                if ts.measurement_uncertainties is not None
                else [],
                "timestamps": ts.measurements.index.strftime("%Y-%m-%d").to_list(),
                "hist_values": hist.tolist(),
                "hist_edges": bin_edges.tolist(),
            }

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
        return out

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
