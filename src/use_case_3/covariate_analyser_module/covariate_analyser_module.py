"""Module for undertaking the covariate analysis."""

from __future__ import annotations

import textwrap
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RANSACRegressor

from use_case_3.covariate_analyser_module.timeseries_container import TimeSeriesContainer
from use_case_3.timeseries_module.timeseries import Timeseries


class CovariateAnalyser:
    """The covariate analysis module."""

    def __init__(
        self: CovariateAnalyser,
        timeseries: TimeSeriesContainer | None,
        alpha: float = 0.05,
    ) -> None:
        """
        Initialise the module by passing in the timeseries container with the processed timeseries for covariate
        analysis.

        Parameters
        ----------
        timeseries : TimeSeriesContainer | None
            The timeseries container.
        alpha : float, optional
            The default significance value used in any hypothesis tests, by default 0.05
        """
        self.alpha = alpha
        self.timeseries = timeseries

    def covariate_analysis(
        self: CovariateAnalyser,
        timeseries_labels: list,
        crosscorr: bool = False,
        axis_labels: list = None,
        max_lag: int = None,
    ) -> tuple[plt.figure, plt.axes]:  # noqa
        """
        Conduct the covariate analysis. This will be the main entrypoint to run the covariate analysis. Currently this
        only runs one method, but the final version will output multiple plots and statistics.

        Parameters
        ----------
        timeseries_labels : list
            The labels of the timeseries which will be included in the covariate analysis.
        crosscorr : bool, optional
            Whether or not the crosscorrelation is used to align the series, by default False
        max_lag : int, optional
            The maximum allowable lag allowed to align the time series. This can be forward or backwards in time, by
            default None
        axis_labels : list, optional
            A list of labels which can be used to customise the axis labels if required, by default None
        """
        joint_grid_plot = self.jointplot_grid(
            timeseries_labels=timeseries_labels, crosscorr=crosscorr, axis_labels=axis_labels, max_lag=max_lag
        )

        return joint_grid_plot

    def correlation_coefficient(
        self: CovariateAnalyser, timeseries_labels: list[str] | None = None, timeseries: list[Timeseries] | None = None
    ) -> tuple[float, float]:
        """
        Calculate the Pearson Correlation Coefficient between two time series. Note only one of either timeseries_labels
        or timeseries can be specified.

        Parameters
        ----------
        timeseries_labels : list[str] | None, optional
            A list of two timeseries labels for which to calculate the correlation coefficient between, by default None
        timeseries : list[Timeseries] | None, optional
            A list of two timeseries for which to calculate the correlation coefficient between, by default None

        Returns
        -------
        tuple[float, float]
            The Pearson correlation coefficient and associated p-value.

        Raises
        ------
        ValueError
            "Both timeseries and timeseries_labels cannot both be specified."
        ValueError
            "One of either timeseries and timeseries_labels must be specified."
        ValueError
            "Only two timeseries labels can be specified"
        ValueError
            "Only two timeseries can be specified"
        """
        if timeseries_labels is not None and timeseries is not None:
            raise ValueError("Both timeseries and timeseries_labels cannot both be specified.")
        elif timeseries_labels is None and timeseries is None:
            raise ValueError("One of either timeseries and timeseries_labels must be specified.")

        if timeseries_labels is not None and len(timeseries_labels) != 2:
            raise ValueError("Only two timeseries labels can be specified")

        if timeseries is not None and len(timeseries) != 2:
            raise ValueError("Only two timeseries can be specified")

        if timeseries_labels is not None:
            timeseries_1 = self.timeseries.get_timeseries([timeseries_labels[0]])[0]
            timeseries_2 = self.timeseries.get_timeseries([timeseries_labels[1]])[0]
        elif timeseries is not None:
            timeseries_1 = timeseries[0]
            timeseries_2 = timeseries[1]

        corr = pearsonr(timeseries_1.measurements, timeseries_2.measurements)

        return corr.statistic, corr.pvalue

    def covariance(
        self: CovariateAnalyser, timeseries_labels: list[str], normalisation: str | None = None
    ) -> pd.DataFrame:
        """
        Calculate the normalised covariance matrix between a set of timeseries.

        Parameters
        ----------
        time_series : list[str]
            The list of timeseries labels to include in the covariance matrix.
        normalisation : normalisation: Union[None, str], optional
            The type of normalisation applied, this can be None, or corrcoef.

        Returns
        -------
        pd.DataFrame
            The covariance matrix.
        """
        if normalisation == "mean_centered":
            timeseries = [
                self.timeseries.get_timeseries([timeseries_label])[0]
                - self.timeseries.get_timeseries([timeseries_label])[0].mean()
                for timeseries_label in timeseries_labels
            ]
        else:
            timeseries = self.timeseries.get_timeseries(timeseries_labels)
        timeseries = [series.measurements for series in timeseries]
        df = pd.concat(timeseries, axis=1)
        if normalisation == "corrcoef":
            return pd.DataFrame(np.corrcoef(df.T), columns=df.columns, index=df.columns)
        else:
            return pd.DataFrame(np.cov(df.T), columns=df.columns, index=df.columns)

    def crosscorrelation(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the cross-correlation between two timeseries.

        Parameters
        ----------
        timeseries_label_1 : str
            The first time series.
        timeseries_label_2 : str
            The second time series.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The cross-correlation with associated lags.
        """
        time_series_1 = (
            self.timeseries.get_timeseries([timeseries_label_1])[0].measurements
            - self.timeseries.get_timeseries([timeseries_label_1])[0].measurements.mean()
        ) / np.std(self.timeseries.get_timeseries([timeseries_label_1])[0].measurements)
        time_series_2 = (
            self.timeseries.get_timeseries([timeseries_label_2])[0].measurements
            - self.timeseries.get_timeseries([timeseries_label_2])[0].measurements.mean()
        ) / np.std(self.timeseries.get_timeseries([timeseries_label_2])[0].measurements)

        return correlate(time_series_1, time_series_2, "full") / min(
            time_series_1.shape[0], time_series_2.shape[0]
        ), correlation_lags(time_series_1.shape[0], time_series_2.shape[0])

    def max_crosscorrelation(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str, max_lag: int | np.int64 | None = None
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
            timeseries_label_1=timeseries_label_1, timeseries_label_2=timeseries_label_2
        )

        if max_lag is None:
            max_lag = max(corr_lags)

        corr = [corr[idx] for idx in range(len(corr_lags)) if np.abs(corr_lags[idx]) <= max_lag]
        corr_lags = [corr_lags[idx] for idx in range(len(corr_lags)) if np.abs(corr_lags[idx]) <= max_lag]

        return corr[np.argmax(np.abs(corr))], corr_lags[np.argmax(np.abs(corr))]

    def _align_timeseries(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str, max_lag: int | None = None
    ) -> tuple[Timeseries, Timeseries, str, int | np.int64]:
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
        tuple[pd.Series, pd.Series, str, int | np.int64]
            The time aligned series according to the maximum cross correlation.
        """
        _, corr_lag = self.max_crosscorrelation(
            timeseries_label_1=timeseries_label_1,
            timeseries_label_2=timeseries_label_2,
            max_lag=max_lag,
        )

        timeseries_1 = deepcopy(self.timeseries.get_timeseries([timeseries_label_1])[0])
        timeseries_2 = deepcopy(self.timeseries.get_timeseries([timeseries_label_2])[0])

        freq = self.timeseries._infer_frequency(timeseries=timeseries_1)

        if freq == "Month":
            timeseries_2.measurements.index = timeseries_2.measurements.index + pd.offsets.MonthEnd(corr_lag)
            if timeseries_2.measurement_uncertainties is not None:
                timeseries_2.measurement_uncertainties.index = (
                    timeseries_2.measurement_uncertainties.index + pd.offsets.MonthEnd(corr_lag)
                )
        elif freq == "Quarter":
            timeseries_2.measurements.index = timeseries_2.measurements.index + pd.offsets.QuarterEnd(corr_lag)
            if timeseries_2.measurement_uncertainties is not None:
                timeseries_2.measurement_uncertainties.index = (
                    timeseries_2.measurement_uncertainties.index + pd.offsets.QuarterEnd(corr_lag)
                )
        elif freq == "Year":
            timeseries_2.measurements.index = timeseries_2.measurements.index + pd.offsets.YearEnd(corr_lag)
            if timeseries_2.measurement_uncertainties is not None:
                timeseries_2.measurement_uncertainties.index = (
                    timeseries_2.measurement_uncertainties.index + pd.offsets.YearEnd(corr_lag)
                )

        start_date, end_date = self.timeseries._find_start_end_dates(
            timeseries_labels=None, timeseries=[timeseries_1, timeseries_2]
        )

        timeseries_trimmed = self.timeseries._trim_timeseries_date(
            timeseries=[timeseries_1, timeseries_2], start_date=start_date, end_date=end_date
        )

        return (
            timeseries_trimmed[0],
            timeseries_trimmed[1],
            freq,
            corr_lag,
        )

    def temporal_offset(
        self: CovariateAnalyser, timeseries_label_1: str, timeseries_label_2: str, metric: str = "mean"
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
            self.timeseries.get_timeseries([timeseries_label_1])[0].measurements
            - self.timeseries.get_timeseries([timeseries_label_2])[0].measurements
        )

        if metric == "mean":
            return diff.mean(), None
        elif metric == "median":
            return diff.median(), None
        elif metric == "max":
            return diff.max(), diff.argmax()
        elif metric == "min":
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
            np.arange(0, self.timeseries.get_timeseries([timeseries_label_1])[0].measurements.shape[0]).reshape(-1, 1),
            self.timeseries.get_timeseries([timeseries_label_1])[0].measurements,
        )
        reg_2 = RANSACRegressor(random_state=0).fit(
            np.arange(0, self.timeseries.get_timeseries([timeseries_label_2])[0].measurements.shape[0]).reshape(-1, 1),
            self.timeseries.get_timeseries([timeseries_label_2])[0].measurements,
        )

        if plot:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].plot(
                np.arange(0, self.timeseries.get_timeseries([timeseries_label_1])[0].measurements.shape[0]).reshape(
                    -1, 1
                ),
                self.timeseries.get_timeseries([timeseries_label_1])[0].measurements,
                ".k",
            )
            axes[0].plot(
                [0, self.timeseries[timeseries_label_1].measurements.shape[0]],
                [
                    reg_1.estimator_.intercept_,
                    (
                        reg_1.estimator_.coef_[0]
                        * self.timeseries.get_timeseries([timeseries_label_1])[0].measurements.shape[0]
                    )
                    + reg_1.estimator_.intercept_,
                ],
                "--k",
            )
            axes[0].grid()
            axes[1].plot(
                np.arange(0, self.timeseries.get_timeseries([timeseries_label_2])[0].measurements.shape[0]).reshape(
                    -1, 1
                ),
                self.timeseries.get_timeseries([timeseries_label_2])[0].measurements,
                ".k",
            )
            axes[1].plot(
                [0, self.timeseries.get_timeseries([timeseries_label_2])[0].measurements.shape[0]],
                [
                    reg_2.estimator_.intercept_,
                    (
                        reg_2.estimator_.coef_[0]
                        * self.timeseries.get_timeseries([timeseries_label_2])[0].measurements.shape[0]
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

    def jointplot_grid(
        self: CovariateAnalyser,
        timeseries_labels: list[str],
        crosscorr: bool = False,
        axis_labels: list = None,
        max_lag: int = None,
    ) -> tuple[plt.figure, plt.axes]:
        """
        Plot a gridded jointplot for the given set of timeseries. Note that you can pass crosscorr True if the maximum
        cross-correlation is desired. This method returns a matplotlib figure and associated axes.

        Parameters
        ----------
        timeseries_labels : list[str]
            The timeseries labels to be plotted.
        crosscorr : bool, optional
            Whether the pearson correlation coefficient (False) or the maximum cross correlation should be used (True),
            by default False.
        max_lag : int, optional
            The maximum allowable lag allowed to align the time series. This can be forward or backwards in time, by
            default None.
        axis_labels : list, optional
            A list of labels which can be used to customise the axis labels if required, by default None.
        """
        timeseries = self.timeseries.get_timeseries(timeseries_labels=timeseries_labels)
        units = self.timeseries.get_units(timeseries_labels=timeseries_labels)

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
                        timeseries_col, timeseries_row, freq, lag = self._align_timeseries(
                            timeseries_labels[col],
                            timeseries_labels[row],
                            max_lag=max_lag,
                        )
                    else:
                        timeseries_col = timeseries[col].copy()
                        timeseries_row = timeseries[row].copy()
                        lag = 0
                        freq = self.timeseries._infer_frequency(timeseries=timeseries_col)

                    if np.abs(lag) != 1:
                        freq = freq + "s"

                    corr = self.correlation_coefficient(timeseries=[timeseries_col, timeseries_row])
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
            timeseries_col.to_frame().columns[0]
            sns.histplot(data=timeseries_row.to_frame(), y=timeseries_row.to_frame().columns[0], ax=axes[row, -1])
            axes[row, -1].spines["top"].set_visible(False)
            axes[row, -1].spines["right"].set_visible(False)
            axes[row, -1].spines["bottom"].set_visible(False)
            axes[row, -1].set_xlabel("")
            axes[row, -1].get_xaxis().set_ticks([])

            sns.histplot(data=timeseries_row.to_frame(), x=timeseries_row.to_frame().columns[0], ax=axes[-1, row])
            axes[-1, row].spines["top"].set_visible(False)
            axes[-1, row].spines["right"].set_visible(False)
            axes[row, -1].spines["bottom"].set_visible(False)
            axes[-1, row].spines["left"].set_visible(False)
            axes[-1, row].set_ylabel("")
            axes[-1, row].get_yaxis().set_ticks([])

            # Add the correct axis labels
            if axis_labels is None:
                axis_label = timeseries_labels[row]
            else:
                axis_label = axis_labels[row]
            label = textwrap.fill(
                " ".join([axis_label, "(" + units[row] + ")"]),
                width=25,
                break_long_words=True,
            )
            axes[row, 0].set_ylabel(label)
            axes[-1, row].set_xlabel(label)

        plt.tight_layout()
        plt.plot()
        return f, axes

    def plot_timeseries(
        self: CovariateAnalyser, timeseries_labels: list = None, yaxis_labels: list = None
    ) -> tuple[plt.figure, plt.axes]:
        """
        Plot a set of timeseries.

        Parameters
        ----------
        timeseries_labels : list[str]
            The timeseries labels to be plotted.
        yaxis_labels : list, optional
            A list of labels which can be used to customise the y-axis labels if required, by default None.
        """
        if timeseries_labels is None:
            timeseries_labels = self.timeseries.get_labels_all()
        f, axes = plt.subplots(
            len(timeseries_labels),
            1,
            figsize=(8, 4 * len(timeseries_labels)),
            sharex="col",
        )

        for n, label in enumerate(timeseries_labels):
            sns.lineplot(
                x=self.timeseries.get_timeseries([timeseries_labels[n]])[0].index,
                y=self.timeseries.get_timeseries([timeseries_labels[n]])[0].values,
                ax=axes[n],
                c="k",
            )
            if yaxis_labels is None:
                yaxis_label = timeseries_labels[n]
            else:
                yaxis_label = yaxis_labels[n]
            axes[n].set_ylabel(yaxis_label + " \n (" + self.timeseries.get_units([label])[0] + ")", fontsize=20)
            axes[n].grid(True)
            axes[n].tick_params(axis="both", which="major", labelsize=15)
            axes[n].tick_params(axis="both", which="minor", labelsize=8)

        axes[n].set_xlabel("Year", fontsize=20)

        return f, axes
