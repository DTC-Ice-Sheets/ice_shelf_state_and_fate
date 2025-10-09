"""Local data processing functions.

Most if not all of this should be migrated to the dtc_query_api in the future.
"""

# Disable linting for this file because it's too much of a mess and most of the code is temporary
# ruff: noqa

from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.dashboard.query_api import config_dict, process_all_data
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesType,
)


def generate_statistic(timeseries_container, variable, steady_state_val=None):
    variable_data = timeseries_container.timeseries[variable].measurements.to_numpy().ravel()
    if timeseries_container.timeseries[variable].measurement_uncertainties is not None:
        variable_unc = timeseries_container.timeseries[variable].measurement_uncertainties.to_numpy().ravel()
    else:
        variable_unc = np.zeros(variable_data.shape)

    total = np.sum(variable_data)

    if steady_state_val is not None:
        total -= steady_state_val * variable_data.shape[0]

    if variable_unc.shape[0] > 0:
        total_unc = 0
        for i in range(variable_unc.shape[0]):
            total_unc = np.sqrt(total_unc**2 + variable_unc[i] ** 2)
    else:
        total_unc = 0

    return total, total_unc


def generate_statistic_cummulative_difference(timeseries_container, variable, year_range):
    variable_data = timeseries_container.timeseries[variable].measurements
    variable_data = variable_data[(variable_data.index >= year_range[0]) & (variable_data.index <= year_range[1])]
    if timeseries_container.timeseries[variable].measurement_uncertainties is not None:
        variable_unc = timeseries_container.timeseries[variable].measurement_uncertainties
        variable_unc = variable_unc[(variable_unc.index >= year_range[0]) & (variable_unc.index <= year_range[1])]
    else:
        variable_unc = np.zeros(variable_data.shape)
    difference = variable_data.iloc[-1] - variable_data.iloc[0]
    difference_unc = np.sqrt(variable_unc.iloc[0] ** 2 + variable_unc.iloc[-1] ** 2) if variable_unc.shape[0] > 0 else 0
    return difference, difference_unc


def generate_statistics(timeseries_container, year_range):
    for label in timeseries_container.timeseries:
        timeseries_container.timeseries[label].summary_statistics()

    # First process the cummulative data
    labels = []
    rates = []
    totals = []
    for label in timeseries_container.timeseries:
        if timeseries_container.timeseries[label].timeseries_type == TimeseriesType.CUMULATIVE:
            labels.append(label)
            rates.append(timeseries_container.timeseries[label].summary_stats.loc["rate of change", "Value"])
            totals.append(timeseries_container.timeseries[label].summary_stats.loc["Difference", "Value"])

    cummulative_summary = (
        pd.DataFrame(zip(labels, rates, totals, strict=False), columns=["Variable", "Rate of change", "Total"])
        .sort_values(["Total"])
        .set_index(["Variable"])
    )

    total_mass_loss = cummulative_summary.loc["cumulative_mass_change", "Total"]
    negative_cummulative_contributors = []
    for idx in cummulative_summary.index:
        if idx != "cumulative_mass_change" and np.abs(cummulative_summary.loc[idx, "Total"] / total_mass_loss) > 0.2:
            negative_cummulative_contributors.append(idx)

    negative_cummulative_contributors = cummulative_summary.loc[negative_cummulative_contributors, :]

    labels = []
    mean = []
    maximum = []
    totals = []
    for label in timeseries_container.timeseries:
        if timeseries_container.timeseries[label].timeseries_type == TimeseriesType.DERIVATIVE:
            labels.append(label)
            mean.append(timeseries_container.timeseries[label].summary_stats.loc["Mean", "Value"])
            maximum.append(timeseries_container.timeseries[label].summary_stats.loc["Maximum Value", "Value"])
            totals.append(timeseries_container.timeseries[label].summary_stats.loc["Total", "Value"])

    derivative_summary = (
        pd.DataFrame(zip(labels, mean, maximum, totals, strict=False), columns=["Variable", "Mean", "Maximum", "Total"])
        .sort_values(["Total"])
        .set_index("Variable")
    )

    return total_mass_loss, negative_cummulative_contributors, derivative_summary


def determine_large_events(dataset: TimeSeriesPreprocessor, variable: str, stationary_check: bool = False):
    if stationary_check:
        dftest = adfuller(dataset.timeseries[variable].measurements.values.ravel(), autolag="AIC")[1]
        stationary = True if dftest <= 0.05 else False
    else:
        dftest = 0
        stationary = True

    data = dataset.timeseries[variable].measurements

    if stationary:
        var_q1 = data.quantile(0.25)
        var_q3 = data.quantile(0.75)
        var_iqr = var_q3 - var_q1

        tukey_upper = var_q3 + (1.5 * var_iqr)
        tukey_lower = var_q1 - (1.5 * var_iqr)

        upper_outlier = data.values.ravel() > tukey_upper
        lower_outlier = data.values.ravel() < tukey_lower

        total_outlier = upper_outlier + lower_outlier
    else:
        upper_outlier = data.values.ravel() > 1e99
        lower_outlier = data.values.ravel() < -1e99

        total_outlier = upper_outlier + lower_outlier

    return total_outlier, upper_outlier, lower_outlier


def dectect_changes(
    dataset: TimeSeriesPreprocessor, variable: str, model: str = "l2", min_size: int = 3, jump: int = 1
):
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(
        dataset.timeseries[variable].measurements.to_numpy().ravel()
    )
    change_brackets = algo.predict(
        pen=50 * np.log(dataset.timeseries[variable].measurements.to_numpy().ravel().shape[0])
    )

    return change_brackets


def calculate_median_change_brackets(dataset: TimeSeriesPreprocessor, variable: str, change_brackets: list[int]):
    change_bracket_prev = 0
    data_var = dataset.timeseries[variable].measurements.to_numpy()
    medians = []
    if len(change_brackets) >= 2:
        for change_bracket in change_brackets:
            medians.append(np.nanmedian(data_var[change_bracket_prev:change_bracket]))
    else:
        medians.append(np.nanmedian(data_var[0:5]))
        medians.append(np.nanmedian(data_var[-5:]))
    return medians


def detect_outlier_events(
    timeseries_container: TimeSeriesPreprocessor,
    variable: str = "calving",
    stationary_check: bool = False,
):
    events_mask, _, _ = determine_large_events(
        dataset=timeseries_container, variable=variable, stationary_check=stationary_check
    )
    variable_data = timeseries_container.timeseries[variable].measurements.to_numpy().ravel()
    if timeseries_container.timeseries[variable].measurement_uncertainties is not None:
        variable_unc = timeseries_container.timeseries[variable].measurement_uncertainties.to_numpy().ravel()
    else:
        variable_unc = np.zeros(variable_data.shape)
    time = timeseries_container.timeseries[variable].measurements.index.ravel()
    variable_events = variable_data[events_mask]
    variable_unc_events = variable_unc[events_mask]
    time_events = time[events_mask]

    idx = np.argsort(np.abs(variable_events))
    variable_events = variable_events[idx]
    variable_unc_events = variable_unc_events[idx]
    time_events = time_events[idx]

    total = np.sum(variable_data)

    if variable_unc.shape[0] > 0:
        total_unc = 0
        for i in range(variable_unc.shape[0]):
            total_unc = np.sqrt(total_unc**2 + variable_unc[i] ** 2)
        # total_unc = np.sqrt(np.sum(np.square(variable_unc))) / variable_unc.shape[0]
    else:
        total_unc = 0

    return total, total_unc, variable_events, variable_unc_events, time_events


def calculate_rate(
    timeseries_container: TimeSeriesPreprocessor,
    variable: str,
    outlier_remove: bool = True,
    change_detection: bool = True,
):
    timeseries_container_process = deepcopy(timeseries_container)

    if outlier_remove:
        outlier_mask, _, _ = determine_large_events(dataset=timeseries_container_process, variable=variable)
        timeseries_container_process.timeseries[variable].measurements[outlier_mask] = np.nan
        timeseries_container_process.timeseries[variable].measurements.interpolate(inplace=True)

    change_idx = dectect_changes(dataset=timeseries_container_process, variable=variable) if change_detection else [0]

    if (
        len(change_idx) > 0
        and (timeseries_container_process.timeseries[variable].measurements.shape[0] - change_idx[-1]) > 4
    ):
        time = timeseries_container_process.timeseries[variable].measurements.index.to_numpy()[change_idx[-1] :]
        measurements = timeseries_container_process.timeseries[variable].measurements.to_numpy()[change_idx[-1] :]
    else:
        time = timeseries_container_process.timeseries[variable].measurements.index.to_numpy()
        measurements = timeseries_container_process.timeseries[variable].measurements.to_numpy()

    return LinearRegression().fit(time.reshape((-1, 1)), measurements.ravel()).coef_[0]


def find_matching_events(events_1, events_2):
    matched_years = []
    for year in events_1[4]:
        if year in events_2[4]:
            matched_years.append(year)

    return matched_years


def calculate_mass_balance(
    timeseries_container: TimeSeriesPreprocessor,
):
    # check calving for large single point changes
    mass_balance_events = detect_outlier_events(
        timeseries_container=timeseries_container, variable="mass_balance", stationary_check=True
    )

    mass_balance_rate = calculate_rate(
        timeseries_container=timeseries_container, variable="mass_balance", outlier_remove=True, change_detection=True
    )

    # check calving for large changes
    bm_flux_events = detect_outlier_events(
        timeseries_container=timeseries_container, variable="basal_melt_flux", stationary_check=True
    )

    # check discharge rate for large changes
    calving_events = detect_outlier_events(
        timeseries_container=timeseries_container, variable="calving", stationary_check=True
    )

    calving_matched_years = find_matching_events(mass_balance_events, calving_events)

    # check basal_melt_flux rate for large changes
    discharge_events = detect_outlier_events(
        timeseries_container=timeseries_container, variable="discharge", stationary_check=True
    )

    # check smb rate for large changes
    smb_events = detect_outlier_events(
        timeseries_container=timeseries_container, variable="surface_mass_balance", stationary_check=True
    )

    return mass_balance_events, mass_balance_rate, calving_matched_years


def example_covariate_analysis_thwaites(
    date_start: datetime,
    date_end: datetime,
) -> tuple[dict, dict, dict]:
    """Perform example covariate analysis for Thwaites ice shelf.

    Parameters
    ----------
    date_start : datetime
        Start date of the analysis.
    date_end : datetime
        End date of the analysis.

    Returns
    -------
    tuple[dict, dict, dict]
        Tuple containing:
        - Dictionary of strongly correlated variable pairs.
        - Dictionary of moderately correlated variable pairs.
        - Dictionary of weakly correlated variable pairs.
    """
    ice_shelf = "thwaites"
    variables = [
        ("moorings_1d", "depth_PIG_N_and_S_combined"),
        ("basal_melt_anomalies_pine_island_1d", "basal_melt_anomalies"),
        ("subglacial_lake_discharge_1d", "lake_discharge"),
        # ("subglacial_lakes_volume_change_1d", "volume_change"),
        ("basal_melt_anomalies_thwaites_1d", "basal_melt_anomalies"),
    ]

    axis_labels = []
    for dataset_id, variable_id in variables:
        axis_labels.append(config_dict[dataset_id]["variables"][variable_id]["y_axis_label"])

    timeseries = process_all_data(
        ice_shelf,
        variables,
        date_start,
        date_end,
    )

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

    strong_correlation = covariate_analyser.find_correlated_vars(corr_thresh=0.8)
    medium_correlation = covariate_analyser.find_correlated_vars(corr_thresh=0.5)
    weak_correlation = covariate_analyser.find_correlated_vars(corr_thresh=0.25)

    for key in strong_correlation:
        del medium_correlation[key]
        del weak_correlation[key]

    for key in medium_correlation:
        del weak_correlation[key]

    return strong_correlation, medium_correlation, weak_correlation


def example_subglacial_lake_analysis(
    ice_shelf: str,
    date_start: datetime,
    date_end: datetime,
) -> tuple[list[str], dict, TimeSeriesPreprocessor]:
    """Perform example analysis of subglacial lakes under an ice shelf.

    Parameters
    ----------
    ice_shelf : str
        Name of the ice shelf to analyse.
    date_start : datetime
        Start date of the analysis.
    date_end : datetime
        End date of the analysis.

    Returns
    -------
    tuple[list[str], dict, TimeSeriesPreprocessor]
        Tuple containing:
        - List of lakes that had events.
        - Dictionary of events for each lake and variable.
        - TimeSeriesPreprocessor object containing the processed time series data.
    """
    variables = [
        ("sub_glacial_lakes", "height_above_wgs84_ellipsoid"),
        ("sub_glacial_lakes", "rate_of_volume_change"),
        ("sub_glacial_lakes", "volume_change"),
    ]

    lakes = [
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW70",
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW124",
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW142",
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW170",
    ]

    timeseries = process_all_data(
        ice_shelf,
        variables,
        date_start,
        date_end,
    )

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
    )

    results = {}

    for lake in lakes:
        results[lake] = {}
        _, _, sg_lake_height_events, _, _ = detect_outlier_events(
            timeseries_container=timeseries_container,
            variable=lake + "-height_above_wgs84_ellipsoid",
        )
        if len(sg_lake_height_events) > 0:
            results[lake]["height_above_wgs84_ellipsoid"] = sg_lake_height_events
        _, _, sg_lake_rate_vol_change_events, _, _ = detect_outlier_events(
            timeseries_container=timeseries_container,
            variable=lake + "-rate_of_volume_change",
        )
        if len(sg_lake_rate_vol_change_events) > 0:
            results[lake]["rate_of_volume_change"] = sg_lake_rate_vol_change_events
        _, _, sg_lake_vol_change_events, _, _ = detect_outlier_events(
            timeseries_container=timeseries_container,
            variable=lake + "-volume_change",
        )
        if len(sg_lake_vol_change_events) > 0:
            results[lake]["volume_change"] = sg_lake_vol_change_events

    lakes_events = []
    for lake in results:
        if len(results[lake]) > 0:
            lakes_events.append(lake.split("_")[-1])

    return lakes_events, results, timeseries_container
