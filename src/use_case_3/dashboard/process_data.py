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
        if dftest <= 0.05:
            stationary = True
        else:
            stationary = False
    else:
        stationary = True

    if stationary:
        data = dataset.timeseries[variable].measurements
    else:
        data = dataset.timeseries[variable].measurements.diff()

    print(variable, data)
    var_q1 = data.quantile(0.25)
    var_q3 = data.quantile(0.75)
    var_iqr = var_q3 - var_q1

    tukey_upper = var_q3 + (1.5 * var_iqr)
    tukey_lower = var_q1 - (1.5 * var_iqr)

    upper_outlier = data.values.ravel() > tukey_upper
    lower_outlier = data.values.ravel() < tukey_lower

    return upper_outlier, lower_outlier


def dectect_changes(dataset: xr.Dataset, variable: str, model: str = "l1", min_size: int = 3, jump: int = 1):
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(dataset[variable].ravel())
    change_brackets = algo.predict(pen=50 * np.log(dataset[variable].ravel().shape[0]))

    return change_brackets


def detect_calving_events(
    timeseries_container: TimeSeriesPreprocessor,
    variable: str = "calving",
):
    calving_events_mask, _ = determine_large_events(dataset=timeseries_container, variable=variable)
    calving_data = timeseries_container.timeseries[variable].measurements.to_numpy().ravel()
    calving_unc = timeseries_container.timeseries[variable].measurement_uncertainties.to_numpy().ravel()
    time = timeseries_container.timeseries[variable].measurements.index.ravel()
    calving_events = calving_data[calving_events_mask]
    calving_unc_events = calving_unc[calving_events_mask]
    time_events = time[calving_events_mask]

    idx = np.argsort(calving_events)
    calving_events = calving_events[idx]
    calving_unc_events = calving_unc_events[idx]
    time_events = time_events[idx]

    if calving_unc.shape[0] > 0:
        total = np.sum(calving_data)
        total_unc = np.sqrt(np.sum(np.square(calving_unc_events))) / calving_unc_events.shape[0]
    else:
        total = 0
        total_unc = 0

    return total, total_unc, calving_events, calving_unc_events, time_events


def calculate_mass_balance(
    timeseries_container: TimeSeriesPreprocessor,
    variable: str = "mass_balance",
):
    mass_balance_events_mask, _ = determine_large_events(dataset=timeseries_container, variable=variable)
    massbalance_data = timeseries_container.timeseries[variable].measurements.to_numpy().ravel()
    massbalance_unc = timeseries_container.timeseries[variable].measurement_uncertainties.to_numpy().ravel()
    time = timeseries_container.timeseries[variable].measurements.index.ravel()

    # check calving for large changes
    determine_large_events(dataset=timeseries_container, variable="basal_melt_flux", stationary_check=True)
    # check discharge rate for large changes
    # check basal_melt_flux rate for large changes
    # check smb rate for large changes
