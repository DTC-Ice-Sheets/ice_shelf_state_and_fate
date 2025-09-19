import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_summary_indicator(summary_data, variable, year_range):
    variable_uncertainty = variable + "_uncertainties"

    summary_data_filt = copy.deepcopy(summary_data)

    summary_data_filt[variable] = summary_data[variable][
        (summary_data["time"] > pd.Timestamp(year=int(year_range[0]), month=1, day=1))
        & (summary_data["time"] < pd.Timestamp(year=int(year_range[1]), month=12, day=31))
    ]

    summary_data_filt[variable_uncertainty] = summary_data[variable_uncertainty][
        (summary_data["time"] >= pd.Timestamp(year=int(year_range[0]), month=1, day=1))
        & (summary_data["time"] <= pd.Timestamp(year=int(year_range[1]), month=12, day=31))
    ]
    var_values = summary_data_filt[variable].values
    var_values = var_values[~np.isnan(var_values)]
    var_values_unc = summary_data_filt[variable_uncertainty].values
    var_values_unc = var_values_unc[~np.isnan(var_values_unc)]

    total_change_value = var_values[-1] - var_values[0]
    uncertainty = var_values_unc[-1]
    uncertainty_upper = total_change_value + uncertainty
    uncertainty_lower = total_change_value - uncertainty

    max_value = abs(summary_data[variable][-1]).values
    if max(abs(uncertainty_upper), abs(uncertainty_lower)) > max_value:
        max_value = max(abs(uncertainty_upper), abs(uncertainty_lower))

    bar_range = [0, total_change_value] if total_change_value >= 0 else [total_change_value, 0]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+delta",
            value=float(total_change_value),
            domain={"x": [0.1, 1], "y": [0, 1]},
            delta={"reference": 0},
            gauge={
                "shape": "bullet",
                "axis": {"range": [-max_value, max_value]},
                "bar": {"thickness": 0},
                "steps": [
                    {
                        "range": [uncertainty_lower, uncertainty_upper],
                        "color": "grey",
                        "thickness": 1,
                    },
                    {
                        "range": bar_range,
                        "color": "black",
                        "thickness": 0.1,
                    },
                ],
            },
        )
    )
    fig.update_layout(height=55)
    fig.update_layout(margin={"t": 0, "l": 0, "b": 25, "r": 0})
    return fig


def plot_gauge_indicator(summary_data, variable):
    variable_uncertainty = variable + "_uncertainty"

    total_change_value = summary_data[variable].values[0][0]
    if variable != "mass_trend":
        uncertainty = summary_data[variable_uncertainty].values[0][0]
    else:
        uncertainty = 0
    uncertainty_upper = total_change_value + uncertainty
    uncertainty_lower = total_change_value - uncertainty

    max_value = abs(total_change_value)
    if max(abs(uncertainty_upper), abs(uncertainty_lower)) > max_value:
        max_value = max(abs(uncertainty_upper), abs(uncertainty_lower))

    if total_change_value >= 0:
        bar_range = [0, total_change_value]
    else:
        bar_range = [total_change_value, 0]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+delta",
            value=float(total_change_value),
            domain={"x": [0.1, 1], "y": [0, 1]},
            delta={"reference": 0},
            gauge={
                "axis": {"range": [-max_value, max_value]},
                "bar": {"thickness": 0},
                "steps": [
                    {
                        "range": [uncertainty_lower, uncertainty_upper],
                        "color": "grey",
                        "thickness": 1,
                    },
                    {
                        "range": bar_range,
                        "color": "black",
                        "thickness": 0.1,
                    },
                ],
            },
        )
    )
    fig.update_layout(height=100)
    fig.update_layout(margin={"t": 10, "l": 25, "b": 10, "r": 25})
    return fig


def plot_summary_timeseries(summary_data, variable, year_range, margin={"t": 0, "l": 0, "b": 25, "r": 0}):
    variable_uncertainty = variable + "_uncertainties"

    summary_data_filt = copy.deepcopy(summary_data)

    summary_data_filt[variable] = summary_data[variable][
        (summary_data["time"] > pd.Timestamp(year=int(year_range[0]), month=1, day=1))
        & (summary_data["time"] < pd.Timestamp(year=int(year_range[1]), month=12, day=31))
    ]

    summary_data_filt[variable_uncertainty] = summary_data[variable_uncertainty][
        (summary_data["time"] >= pd.Timestamp(year=int(year_range[0]), month=1, day=1))
        & (summary_data["time"] <= pd.Timestamp(year=int(year_range[1]), month=12, day=31))
    ]
    var_values = summary_data_filt[variable].values.ravel()
    var_mask = ~np.isnan(var_values)
    var_values = var_values[var_mask]
    var_values_unc = summary_data_filt[variable_uncertainty].values.ravel()
    var_values_unc = var_values_unc[var_mask]

    time = summary_data_filt["time"].values
    time = time[var_mask]

    total_change_value = var_values[-1] - var_values[0]
    uncertainty = var_values_unc[-1]
    uncertainty_upper = total_change_value + uncertainty
    uncertainty_lower = total_change_value - uncertainty

    max_value = abs(summary_data[variable][-1]).values
    if max(abs(uncertainty_upper), abs(uncertainty_lower)) > max_value:
        max_value = max(abs(uncertainty_upper), abs(uncertainty_lower))

    fig = go.Figure(
        go.Scatter(x=time, y=var_values, mode="lines", fill=None, line_color="black"),
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=var_values + var_values_unc,
            mode="lines",
            line_color="rgba(0,0,0,0.5)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=var_values - var_values_unc,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(0,0,0,0.2)",
            line_color="rgba(0,0,0,0.5)",
        ),
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(height=150)
    fig.update_layout(margin=margin)

    return fig
