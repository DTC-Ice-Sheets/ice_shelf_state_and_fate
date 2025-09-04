import plotly.graph_objects as go
import textwrap
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import copy
import numpy as np


def plot_summary_indicator(summary_data, variable, year_range):
    variable_uncertainty = variable + "_uncertainties"
    variable_unit = "Gt"

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

    max_value = abs(summary_data_filt[variable][-1]).values
    if max(abs(uncertainty_upper), abs(uncertainty_lower)) > max_value:
        max_value = max(abs(uncertainty_upper), abs(uncertainty_lower))

    chart_title = "<br>".join(textwrap.wrap(f"<b>{' '.join(variable.split('_'))} ({variable_unit})</b>", width=20))

    if total_change_value >= 0:
        bar_range = [0, total_change_value]
    else:
        bar_range = [total_change_value, 0]
    fig = go.Figure(
        go.Indicator(
            mode="number+gauge+delta",
            value=float(total_change_value),
            domain={"x": [0.1, 1], "y": [0, 1]},
            title={"text": chart_title},
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
    fig.update_layout(height=150)
    fig.update_layout(margin={"t": 50, "l": 200, "b": 35, "r": 75})
    return fig
