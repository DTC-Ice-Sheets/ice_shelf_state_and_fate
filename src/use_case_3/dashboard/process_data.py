from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    NormalisationType,
    TimeseriesRegion,
    TimeseriesType,
    UncertaintyType,
)

import numpy as np
import pandas as pd
import xarray as xr
import ruptures as rpt
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from copy import deepcopy

from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor


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

    print(cummulative_summary)

    negative_cummulative_contributors = cummulative_summary.loc[negative_cummulative_contributors, :]
    print(negative_cummulative_contributors)

    labels = []
    mean = []
    maximum = []
    totals = []
    for label in timeseries_container.timeseries:
        if timeseries_container.timeseries[label].timeseries_type == TimeseriesType.DERIVATIVE:
            labels.append(label)
            # print(label)
            mean.append(timeseries_container.timeseries[label].summary_stats.loc["Mean", "Value"])
            maximum.append(timeseries_container.timeseries[label].summary_stats.loc["Maximum Value", "Value"])
            totals.append(timeseries_container.timeseries[label].summary_stats.loc["Total", "Value"])

    derivative_summary = (
        pd.DataFrame(zip(labels, mean, maximum, totals, strict=False), columns=["Variable", "Mean", "Maximum", "Total"])
        .sort_values(["Total"])
        .set_index("Variable")
    )

    print(derivative_summary)

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
    dataset: TimeSeriesPreprocessor, variable: str, model: str = "l1", min_size: int = 3, jump: int = 1
):
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(
        dataset.timeseries[variable].measurements.to_numpy().ravel()
    )
    change_brackets = algo.predict(
        pen=50 * np.log(dataset.timeseries[variable].measurements.to_numpy().ravel().shape[0])
    )

    return change_brackets


def detect_outlier_events(
    timeseries_container: TimeSeriesPreprocessor,
    variable: str = "calving",
    stationary_check: bool = False,
):
    events_mask, _, _ = determine_large_events(
        dataset=timeseries_container, variable=variable, stationary_check=stationary_check
    )
    variable_data = timeseries_container.timeseries[variable].measurements.to_numpy().ravel()
    variable_unc = timeseries_container.timeseries[variable].measurement_uncertainties.to_numpy().ravel()
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
    matched_values = []
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
