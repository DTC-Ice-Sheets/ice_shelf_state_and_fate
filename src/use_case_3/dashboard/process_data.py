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
