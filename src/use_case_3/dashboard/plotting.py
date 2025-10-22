"""Dashboard plotting functions."""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries import Timeseries


def plot_gauge_indicator_timeseries(timeseries_container: TimeSeriesPreprocessor, variable: str) -> go.Figure:
    """
    Plot a gauge indicator for the total change in a timeseries variable.

    Parameters
    ----------
    timeseries_container : TimeSeriesPreprocessor
        The container holding the timeseries data.
    variable : str
        The variable to plot.

    Returns
    -------
    go.Figure
        A Plotly figure containing the gauge indicator.
    """
    measurements = timeseries_container.timeseries[variable].measurements.to_numpy()
    measurement_unc = timeseries_container.timeseries[variable].measurement_uncertainties.to_numpy()
    total_change_value = np.sum(measurements)

    if measurement_unc is not None:
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

    bar_range = [0, total_change_value] if total_change_value >= 0 else [total_change_value, 0]

    fig = go.Figure(
        go.Indicator(
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


def plot_summary_timeseries(
    timeseries: Timeseries,
    outliers: tuple[np.ndarray, np.ndarray] | None = None,
    brackets: list[int] | None = None,
    y_label: str | None = None,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """Plot a timeseries object.

    Parameters
    ----------
    timeseries : Timeseries
        The timeseries object to plot.
    outliers : tuple[np.ndarray, np.ndarray] | None, optional
        An optional tuple of (x, y) coordinates for outliers to plot.
    brackets : list[int] | None, optional
        An optional list of indices to draw vertical brackets at.
    y_label : str | None, optional
        An optional label for the y-axis.
    margin : dict[str, int] | None, optional
        An optional dictionary specifying the margins of the plot.

    Returns
    -------
    go.Figure
        A Plotly figure containing the timeseries plot.
    """
    if margin is None:
        margin = {"t": 0, "l": 50, "b": 25, "r": 200}
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
    for i, bracket in enumerate(brackets or []):
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

    if outliers is not None:
        fig.add_trace(
            go.Scatter(
                x=outliers[0],
                y=outliers[1],
                mode="markers",
                marker={"color": "red", "size": 10, "line": {"color": "black", "width": 2}},
            ),
        )

    fig.update_layout(showlegend=False)
    fig.update_layout(height=250)
    fig.update_layout(margin=margin)

    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)

    return fig


def plot_summary_timeseries_multiple(
    timeseries_container: TimeSeriesPreprocessor,
    variables: list[str],
    outliers: tuple[np.ndarray, np.ndarray] | None = None,
    y_label: str | None = None,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """Plot multiple timeseries from a timeseries container.

    Parameters
    ----------
    timeseries_container : TimeSeriesPreprocessor
        The container holding the timeseries data.
    variables : list[str]
        The list of variable names to plot.
    outliers : tuple[np.ndarray, np.ndarray] | None, optional
        An optional tuple of (x, y) coordinates for outliers to plot.
    y_label : str | None, optional
        An optional label for the y-axis.
    margin : dict[str, int] | None, optional
        An optional dictionary specifying the margins of the plot.

    Returns
    -------
    go.Figure
        A Plotly figure containing the timeseries plot.
    """
    if margin is None:
        margin = {"t": 0, "l": 50, "b": 25, "r": 200}
    colours = ["rgba(0,0,0,1)", "rgba(0,0,255,1)", "rgba(255,0,0,1)", "rgba(0,255,0,1)", "rgba(0,0,0,1)"]
    colours_fill = ["rgba(0,0,0,0.1)", "rgba(0,0,255,0.1)", "rgba(255,0,0,0.1)", "rgba(0,255,0,0.1)", "rgba(0,0,0,0.1)"]
    fig = None
    for i, variable in enumerate(variables):
        time = timeseries_container.timeseries[variable].measurements.index
        measurements = timeseries_container.timeseries[variable].measurements.values

        if timeseries_container.timeseries[variable].measurement_uncertainties is not None:
            uncertainty = timeseries_container.timeseries[variable].measurement_uncertainties.values
        else:
            uncertainty = measurements

        lake = variable.split("-")[0].split("_")[-1]

        if fig is None:
            fig = go.Figure(
                go.Scatter(x=time, y=measurements, mode="lines", fill=None, line_color=colours[i], name=lake),
            )
        else:
            fig.add_trace(
                go.Scatter(x=time, y=measurements, mode="lines", line_color=colours[i], name=lake),
            )

        fig.add_trace(
            go.Scatter(
                x=time,
                y=measurements + uncertainty,
                mode="lines",
                line_color="rgba(0,0,0,0.25)",
                showlegend=False,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=time,
                y=measurements - uncertainty,
                mode="lines",
                fill="tonexty",
                fillcolor=colours_fill[i],
                line_color="rgba(0,0,0,0.25)",
                showlegend=False,
            ),
        )

        if outliers is not None:
            fig.add_trace(
                go.Scatter(
                    x=outliers[0],
                    y=outliers[1],
                    mode="markers",
                    marker={"color": "red", "size": 10, "line": {"color": "black", "width": 2}},
                    showlegend=False,
                ),
            )
    assert fig is not None, "No variables to plot"

    fig.update_layout(showlegend=True)
    fig.update_layout(height=250)
    fig.update_layout(margin=margin)

    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)

    return fig


def plot_covariate_analysis(plot_data: dict) -> go.Figure:
    """
    Plot the results of a covariate analysis.

    Parameters
    ----------
    plot_data : dict
        Dictionary containing the data for plotting.

    Returns
    -------
    go.Figure
        A Plotly figure containing the covariate analysis results.
    """
    colors = qualitative.Plotly
    labels = list(plot_data["variables"].keys())
    datas = [plot_data["variables"][label] for label in labels]
    titles = [d["title"] for d in datas] + ["Frequency"]
    fig = make_subplots(
        len(labels) + 1,
        len(labels) + 1,
        row_titles=titles,
        column_titles=titles,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
    )

    for idx, data in enumerate(datas):
        edge_starts = data["hist_edges"][:-1]
        edge_ends = data["hist_edges"][1:]
        widths = [b - a for a, b in zip(edge_starts, edge_ends, strict=True)]
        y = data["hist_values"]
        fig.add_trace(
            go.Bar(
                x=edge_starts,
                customdata=edge_ends,
                width=widths,
                y=y,
                offset=0,
                orientation="v",
                marker={"color": colors[1]},
                hovertemplate=f"<b>%{{x:.2f}} to %{{customdata:.2f}}: </b> %{{y}}<extra>{titles[idx]}</extra>",
            ),
            row=len(labels) + 1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Bar(
                y=edge_starts,
                customdata=edge_ends,
                width=widths,
                x=y,
                offset=0,
                orientation="h",
                marker={"color": colors[1]},
                hovertemplate=f"<b>%{{y:.2f}} to %{{customdata:.2f}}: </b> %{{x}}<extra>{titles[idx]}</extra>",
            ),
            row=idx + 1,
            col=len(labels) + 1,
        )

    for a_idx, a_label in enumerate(labels):
        for b_idx, b_label in enumerate(labels):
            if a_idx == b_idx:
                continue
            a_data, b_data = (
                plot_data["variables"][a_label],
                plot_data["variables"][b_label],
            )
            fig.add_trace(
                go.Scatter(
                    x=a_data["values"],
                    y=b_data["values"],
                    mode="markers",
                    marker={"color": colors[0]},
                    hovertemplate=(
                        f"<b>{titles[a_idx]}:</b> %{{x:.2f}}<br><b>{titles[b_idx]}:</b> %{{y:.2f}}<extra></extra>"
                    ),
                ),
                row=b_idx + 1,
                col=a_idx + 1,
            )
            stats = plot_data["covariate_stats"].get(
                f"{a_label}/{b_label}",
                plot_data["covariate_stats"].get(
                    f"{b_label}/{a_label}",
                    {},
                ),
            )
            axes_id = a_idx + b_idx * (len(labels) + 1) + 1
            axes_name = str(axes_id) if axes_id > 0 else ""
            fig.add_annotation(
                text="<br>".join(
                    [f"<b>{k}:</b> {round(v, 2) if isinstance(v, float) else v}" for k, v in stats.items()]
                ),
                xref=f"x{axes_name} domain",
                yref=f"y{axes_name} domain",
                x=0.01,
                y=0.99,
                showarrow=False,
                align="left",
                bgcolor="white",
                borderpad=5,
                opacity=0.9,
            )

    for i in range(len(labels) + 1):
        fig.add_trace(go.Scatter(), row=i + 1, col=i + 1)

    fig.update_layout(
        height=1024,
        width=1024,
        showlegend=False,
    )
    return fig


def plot_dataset_summary(plot_data: dict) -> go.Figure:
    """Plot the output of the dataset summary module.

    Parameters
    ----------
    plot_data : dict
        The dataset summary data to plot. See `dataset_summary_module.py` for details.

    Returns
    -------
    go.Figure
        A Plotly figure containing the dataset summary plot.
    """
    fig = go.Figure()
    for name, var in plot_data["vars"].items():
        values = np.array(var["values"])
        if "x" not in var["coords"] or "y" not in var["coords"]:
            continue
        xi = var["coords"].index("x")
        yi = var["coords"].index("y")
        mean_vals = values.mean(axis=tuple(i for i in range(values.ndim) if i not in (xi, yi)), keepdims=False)
        fig.add_trace(go.Heatmap(z=mean_vals, name=name))
    return fig
