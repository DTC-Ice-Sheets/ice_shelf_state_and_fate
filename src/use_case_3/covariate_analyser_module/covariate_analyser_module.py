"""Module for undertaking the covariate analysis."""

from __future__ import annotations

import textwrap
from copy import deepcopy
from enum import Enum
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

from use_case_3.covariate_analyser_module.timeseries_preprocessor import AggregationStrategy, TimeSeriesPreprocessor


class ModelEvaluationMetric(Enum):
    """The available model evaluation metrics."""

    AIC = "aic"
    BIC = "bic"

    def __init__(self, evaluation_metric: str) -> None:
        self.evaluation_metric = evaluation_metric


class JointPlotAnalysisType(Enum):
    """The available analysis types used to evaluate correlation or causality between variables."""

    CORR = "corr"
    CROSS_CORR = "cross_corr"
    CAUSAL = "causal"

    def __init__(self, analysis_type: str) -> None:
        self.analysis_type = analysis_type


class CovariateAnalyser:
    """The covariate analysis module."""

    def __init__(
        self: CovariateAnalyser,
        timeseries_container: TimeSeriesPreprocessor,
        alpha: float = 0.05,
    ) -> None:
        """
        Initialise the module by passing in the timeseries container with the processed timeseries for covariate
        analysis.

        TODO: None of the plotting functions are tested!

        Parameters
        ----------
        timeseries : TimeSeriesPreprocessor | None
            The timeseries container.
        alpha : float, optional
            The default significance value used in any hypothesis tests, by default 0.05
        """
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

    def covariate_analysis(
        self: CovariateAnalyser,
        axis_labels: list = None,
        max_lag: int = None,
        analysis_type: JointPlotAnalysisType = JointPlotAnalysisType.CORR,
    ) -> tuple[plt.figure, plt.axes]:  # noqa
        """
        Conduct the covariate analysis. This will be the main entrypoint to run the covariate analysis. Currently this
        only runs one method, but the final version will output multiple plots and statistics.

        Parameters
        ----------
        crosscorr : bool, optional
            Whether or not the crosscorrelation is used to align the series, by default False
        max_lag : int, optional
            The maximum allowable lag allowed to align the time series. This can be forward or backwards in time, by
            default None
        axis_labels : list, optional
            A list of labels which can be used to customise the axis labels if required, by default None
        """
        joint_grid_plot = self.jointplot_grid(axis_labels=axis_labels, max_lag=max_lag, analysis_type=analysis_type)

        return joint_grid_plot

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
            if (
                timeseries_label_1 not in self.timeseries_container._timeseries_altered.keys()
                or timeseries_label_2 not in self.timeseries_container._timeseries_altered.keys()
            ):
                raise ValueError("The labels specified are not in the altered timeseries container")
            timeseries_1 = self.timeseries_container._timeseries_altered[timeseries_label_1]
            timeseries_2 = self.timeseries_container._timeseries_altered[timeseries_label_2]
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
        time_series : list[str]
            The list of timeseries labels to include in the covariance matrix.
        normalisation : normalisation: Union[None, str], optional
            The type of normalisation applied, this can be None, or corrcoef.
        altered : bool, optional
            A boolean flag specifying whether the original timeseries or the temporarily altered timeseries are used.

        Returns
        -------
        pd.DataFrame
            The covariance matrix.
        """
        if altered:
            timeseries = [
                ts for ts in self.timeseries_container._timeseries_altered.values() if ts.label in timeseries_labels
            ]
        else:
            timeseries = [ts for ts in self.timeseries_container.timeseries.values() if ts.label in timeseries_labels]

        if normalisation == "mean_centered":
            timeseries = [ts.measurements - ts.measurements.mean() for ts in timeseries]
        else:
            timeseries = [ts.measurements for ts in timeseries]

        df = pd.concat(timeseries, axis=1)
        if normalisation == "corrcoef":
            return pd.DataFrame(np.corrcoef(df.T), columns=df.columns, index=df.columns)
        else:
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
        altered : bool, optional
            A boolean flag specifying whether the original timeseries or the temporarily altered timeseries are used.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The cross-correlation with associated lags.
        """
        if altered:
            timeseries_1 = deepcopy(self.timeseries_container._timeseries_altered[timeseries_label_1])
            timeseries_2 = deepcopy(self.timeseries_container._timeseries_altered[timeseries_label_2])
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
        max_lag : Union[int, None], optional
            The maximum lag to consider, not this will apply this maximum lag in both the positive and the negative
            directions. If set to None there is no maximum lag, by default None

        Returns
        -------
        tuple[float, int]
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
    ) -> tuple[str, int | np.int64]:
        """
        Align two timeseries according to the maximum cross correlation.

        Parameters
        ----------
        timeseries_label_1 : str
            The label for the first timeseries.
        timeseries_label_2 : str
            The label for the second timeseries.
        max_lag : int | None, optional
            The maximum lag to consider, not this will apply this maximum lag in both the positive and the negative
            directions. If set to None there is no maximum lag, by default None

        Returns
        -------
        tuple[str, int | np.int64]
            The time aligned series according to the maximum cross correlation.
        """
        _, corr_lag = self.max_crosscorrelation(
            timeseries_label_1=timeseries_label_1,
            timeseries_label_2=timeseries_label_2,
            max_lag=max_lag,
        )

        timeseries_1 = deepcopy(self.timeseries_container.timeseries[timeseries_label_1])
        timeseries_2 = deepcopy(self.timeseries_container.timeseries[timeseries_label_2])

        freq = self.timeseries_container.aggregation_period.time_period_shorthand

        if freq == "M":
            timeseries_2.measurements.index = timeseries_2.measurements.index + pd.offsets.MonthEnd(corr_lag)
            if timeseries_2.measurement_uncertainties is not None:
                timeseries_2.measurement_uncertainties.index = (
                    timeseries_2.measurement_uncertainties.index + pd.offsets.MonthEnd(corr_lag)
                )
        elif freq == "Q":
            timeseries_2.measurements.index = timeseries_2.measurements.index + pd.offsets.QuarterEnd(corr_lag)
            if timeseries_2.measurement_uncertainties is not None:
                timeseries_2.measurement_uncertainties.index = (
                    timeseries_2.measurement_uncertainties.index + pd.offsets.QuarterEnd(corr_lag)
                )
        elif freq == "Y":
            timeseries_2.measurements.index = timeseries_2.measurements.index + pd.offsets.YearEnd(corr_lag)
            if timeseries_2.measurement_uncertainties is not None:
                timeseries_2.measurement_uncertainties.index = (
                    timeseries_2.measurement_uncertainties.index + pd.offsets.YearEnd(corr_lag)
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

            self.timeseries_container._timeseries_altered[ts.label] = ts

        return (
            freq,
            corr_lag,
        )

    def temporal_offset(
        self: CovariateAnalyser,
        timeseries_label_1: str,
        timeseries_label_2: str,
        metric: AggregationStrategy = AggregationStrategy.MEAN,
    ) -> tuple[float | int | np.float64 | np.int64, np.ndarray | int | np.int64 | None]:
        """
        Calculate a temporal offset. This is the difference between two series. This can be done using various
        statistics such as the mean, median, maximum and minimum.

        Parameters
        ----------
        timeseries_label_1 : str
            The label for the first timeseries.
        timeseries_label_2 : str
            The label for the second timeseries.
        metric : str, optional
            The statistic to quantify the difference between the two time series, by default "mean".

        Returns
        -------
        tuple[float | int | np.float64 | np.int64, np.ndarray | int | np.int64 | None]
            The temporal offset and if applicable the associated index.
        """
        diff = (
            self.timeseries_container.timeseries[timeseries_label_1].measurements
            - self.timeseries_container.timeseries[timeseries_label_2].measurements
        )

        if metric == AggregationStrategy.MEAN:
            return diff.mean(), None
        elif metric == AggregationStrategy.MEDIAN:
            return diff.median(), None
        elif metric == AggregationStrategy.MAX:
            return diff.max(), diff.argmax()
        elif metric == AggregationStrategy.MIN:
            return diff.min(), diff.argmin()

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
            fig, axes = plt.subplots(nrows=1, ncols=2)
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
        """
        reg_1 = LinearRegression().fit(
            self.timeseries.get_timeseries([timeseries_label_1])[0].values.reshape(-1, 1),
            self.timeseries.get_timeseries([timeseries_label_2])[0].values.reshape(-1, 1),
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

    def jointplot_grid(  # noqa C901
        self: CovariateAnalyser,
        axis_labels: list = None,
        max_lag: int = None,
        analysis_type: JointPlotAnalysisType = JointPlotAnalysisType.CORR,
    ) -> tuple[plt.figure, plt.axes]:
        """
        Plot a gridded jointplot for the given set of timeseries. Note that you can pass crosscorr True if the maximum
        cross-correlation is desired. This method returns a matplotlib figure and associated axes.

        TODO: This is currently a fairly horrible method, but not sure how to simplify currently.

        Parameters
        ----------
        timeseries_labels : list[str]
            The timeseries labels to be plotted.
        max_lag : int, optional
            The maximum allowable lag allowed to align the time series. This can be forward or backwards in time, by
            default None.
        axis_labels : list, optional
            A list of labels which can be used to customise the axis labels if required, by default None.
        analysis_type : JointPlotAnalysisType, optional
            Define the analysis type to be conducted.
        """
        timeseries = list(self.timeseries_container.timeseries.values())

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
                    if analysis_type == JointPlotAnalysisType.CROSS_CORR:
                        freq, lag = self._align_timeseries(
                            timeseries[col].label,
                            timeseries[row].label,
                            max_lag=max_lag,
                        )
                        timeseries_col = deepcopy(self.timeseries_container._timeseries_altered[timeseries[col].label])
                        timeseries_row = deepcopy(self.timeseries_container._timeseries_altered[timeseries[row].label])
                        altered = True
                    elif analysis_type == JointPlotAnalysisType.CORR:
                        timeseries_col = deepcopy(timeseries[col])
                        timeseries_row = deepcopy(timeseries[row])
                        lag = 0
                        freq = self.timeseries_container.aggregation_period.time_period_shorthand
                        altered = False
                    elif analysis_type == JointPlotAnalysisType.CAUSAL:
                        timeseries_col = deepcopy(timeseries[col])
                        timeseries_row = deepcopy(timeseries[row])
                        lag = 0
                        freq = self.timeseries_container.aggregation_period.time_period_shorthand
                        altered = False
                        _, p_value = self._granger_causality_test(
                            timeseries_label_1=timeseries_col.label,
                            timeseries_label_2=timeseries_row.label,
                            max_lags=max_lag,
                            metric=ModelEvaluationMetric.BIC,
                        )
                    if np.abs(lag) != 1:
                        freq = freq + "s"

                    sns.regplot(x=timeseries_col.measurements, y=timeseries_row.measurements, ax=axes[row, col])
                    axes[row, col].set_xlabel("")
                    axes[row, col].set_ylabel("")
                    axes[row, col].grid()

                    if analysis_type in [JointPlotAnalysisType.CORR, JointPlotAnalysisType.CROSS_CORR]:
                        corr, p_value = self.correlation_coefficient(
                            timeseries_label_1=timeseries_col.label,
                            timeseries_label_2=timeseries_row.label,
                            altered=altered,
                        )
                        title = f"corr:{corr:.2f}, p:{p_value:.2f}"
                    elif analysis_type == JointPlotAnalysisType.CAUSAL:
                        title = f"p:{p_value:.2f}"

                    if p_value > self.alpha:
                        axes[row, col].set_facecolor("mistyrose")

                    y_offset = 0.85
                    if analysis_type == JointPlotAnalysisType.CROSS_CORR:
                        title += f", \n lag:{lag} {freq}"
                        y_offset = 0.75
                    axes[row, col].set_title(title, y=y_offset)
                else:
                    axes[row, col].set_facecolor("lightgrey")

            # plot the histograms
            timeseries_col.measurements.to_frame().columns[0]
            sns.histplot(
                data=timeseries_row.measurements.to_frame(),
                y=timeseries_row.measurements.to_frame().columns[0],
                ax=axes[row, -1],
            )
            axes[row, -1].spines["top"].set_visible(False)
            axes[row, -1].spines["right"].set_visible(False)
            axes[row, -1].spines["bottom"].set_visible(False)
            axes[row, -1].set_xlabel("")
            axes[row, -1].get_xaxis().set_ticks([])

            sns.histplot(
                data=timeseries_row.measurements.to_frame(),
                x=timeseries_row.measurements.to_frame().columns[0],
                ax=axes[-1, row],
            )
            axes[-1, row].spines["top"].set_visible(False)
            axes[-1, row].spines["right"].set_visible(False)
            axes[row, -1].spines["bottom"].set_visible(False)
            axes[-1, row].spines["left"].set_visible(False)
            axes[-1, row].set_ylabel("")
            axes[-1, row].get_yaxis().set_ticks([])

            # Add the correct axis labels
            if axis_labels is None:
                axis_label = timeseries_row.label
            else:
                axis_label = axis_labels[row]
            label = textwrap.fill(
                " ".join([axis_label, "(" + timeseries_row.unit + ")"]),
                width=25,
                break_long_words=True,
            )
            axes[row, 0].set_ylabel(label)
            axes[-1, row].set_xlabel(label)

        plt.tight_layout()
        plt.plot()
        return f, axes

    def plot_timeseries(self: CovariateAnalyser, yaxis_labels: list = None) -> tuple[plt.figure, plt.axes]:
        """
        Plot all the timeseries in the timeseries container.

        Parameters
        ----------
        yaxis_labels : list, optional
            A list of labels which can be used to customise the y-axis labels if required, by default None.
        """
        timeseries_labels = list(self.timeseries_container.timeseries)
        f, axes = plt.subplots(
            len(timeseries_labels),
            1,
            figsize=(10, 4 * len(timeseries_labels)),
            sharex="col",
        )

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
                    y=self.timeseries_container.timeseries[timeseries_labels[n]].measurements.values
                    + self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.values,
                    ax=axes[n],
                    c=self.colours[n],
                    linestyle="--",
                    alpha=0.75,
                )
                sns.lineplot(
                    x=self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.index,
                    y=self.timeseries_container.timeseries[timeseries_labels[n]].measurements.values
                    - self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.values,
                    ax=axes[n],
                    c=self.colours[n],
                    linestyle="--",
                    alpha=0.75,
                )
                axes[n].fill_between(
                    self.timeseries_container.timeseries[timeseries_labels[n]].measurements.index,
                    self.timeseries_container.timeseries[timeseries_labels[n]].measurements.values
                    + self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.values,
                    self.timeseries_container.timeseries[timeseries_labels[n]].measurements.values
                    - self.timeseries_container.timeseries[timeseries_labels[n]].measurement_uncertainties.values,
                    alpha=0.5,
                    facecolor=self.colours[n],
                )
            if yaxis_labels is None:
                yaxis_label = timeseries_labels[n]
            else:
                yaxis_label = yaxis_labels[n]
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
        metric : ModelEvaluationMetric, optional
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
        best_bic = 1e99

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
        Conduct a granger causality test to determine causality between two variables. This finds the lag (i.e. the
        number of lag variables in time for both X and Y) which results in the best model fit. Note, that in this case
        we are checking whether time_series_label_1 can be used to forecast time_series_2.

        Parameters
        ----------
        timeseries_label_1 : str
            The label of the first time series (X)
        timeseries_label_2 : str
            The variable for the second time series.
        max_lags : int
            The maximum number of lags, if to large will be adjusted to floor((n - 1) / 3) - 1 where n is the no. rows
             in data.
        metric : ModelEvaluationMetric, optional
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

        if max_lags > max_lags_total:
            max_lags = max_lags_total

        if max_lags < 1:
            raise ValueError("The number of lags must be greater than 0")

        # Iteratively fit the data and determine the model with the lowest error
        best_lag, _ = self._granger_causality_test_bestlag(data=data, max_lags=max_lags, metric=metric)

        gc_res = grangercausalitytests(data, [best_lag], verbose=False)

        return gc_res[best_lag][0]["ssr_ftest"][0], gc_res[best_lag][0]["ssr_ftest"][1]
