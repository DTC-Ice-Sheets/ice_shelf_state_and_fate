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


def plot_gauge_indicator_timeseries(timeseries_container, variable):
    measurements = timeseries_container.timeseries[variable].measurements.to_numpy()
    measurement_unc = timeseries_container.timeseries[variable].measurement_uncertainties.to_numpy()
    total_change_value = np.sum(measurements)

    # if variable in ["basal_melt_flux", "calving"]:
    #     total_change_value = total_change_value * -1

    if measurement_unc is not None:
        # uncertainty = np.sum(measurement_unc)
        uncertainty = 0
        for i in range(measurement_unc.shape[0]):
            uncertainty = np.sqrt(uncertainty**2 + measurement_unc[i] ** 2)
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
            # mode="gauge",
            mode="gauge+number",
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


def plot_summary_timeseries(
    timeseries, outliers=(), brackets=[], y_label=None, margin={"t": 0, "l": 50, "b": 25, "r": 200}
):
    time = timeseries.measurements.index
    measurements = timeseries.measurements.values

    if timeseries.measurement_uncertainties is not None:
        uncertainty = timeseries.measurement_uncertainties.values
    else:
        uncertainty = measurements

    fig = go.Figure(
        go.Scatter(x=time, y=measurements, mode="lines", fill=None, line_color="black"),
    )

    bracket_prev = time[0]
    for i, bracket in enumerate(brackets):
        fillcolor = "grey" if i % 2 == 0 else "white"
        if bracket == time.shape[0]:
            bracket -= 1
        fig.add_vrect(
            x0=bracket_prev,
            x1=time[bracket],
            fillcolor=fillcolor,
            opacity=0.2,
            line_width=0,
        )
        bracket_prev = time[bracket]

    fig.add_trace(
        go.Scatter(
            x=time,
            y=measurements + uncertainty,
            mode="lines",
            line_color="rgba(0,0,0,1)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=measurements - uncertainty,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(211,211,211,1)",
            line_color="rgba(0,0,0,1)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=measurements,
            mode="lines",
            line_color="rgba(0,0,0,1)",
        ),
    )

    if len(outliers) > 0:
        fig.add_trace(
            go.Scatter(
                x=outliers[0],
                y=outliers[1],
                mode="markers",
                marker=dict(color="red", size=10, line=dict(color="black", width=2)),
            ),
        )

    fig.update_layout(showlegend=False)
    fig.update_layout(height=250)
    fig.update_layout(margin=margin)

    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)

    return fig
