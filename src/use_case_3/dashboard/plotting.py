"""Dashboard plotting functions."""

import textwrap
from collections import defaultdict
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import pyproj
from dash import html
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from shapely import MultiPolygon
from shapely.ops import transform

from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import JointPlotAnalysisType


def plot_placeholder() -> go.Figure:
    """Create a clean placeholder for a figure that will be created later.

    Returns
    -------
    go.Figure
        The placeholder figure.
    """
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data to display",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 28},
                "x": 0.5,
                "y": 0.5,
            }
        ],
    )
    return fig


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


def analyse_correlation_strength(plot_data: dict, analysis_type: JointPlotAnalysisType) -> tuple[dict, dict, dict]:
    """
    Analyse the strength of correlations in the plot data.

    Parameters
    ----------
    plot_data : dict
        Dictionary containing the data for plotting.
    analysis_type : JointPlotAnalysisType
        The type of correlation to analyse

    Returns
    -------
    tuple[dict, dict, dict]
        Three dictionaries containing strong, medium, and weak correlations.
    """
    strong_correlation = 0.8
    medium_correlation = 0.5
    weak_correlation = 0.2

    strong_correlations = {}
    medium_correlations = {}
    weak_correlations = {}
    for variables in plot_data["covariate_stats"]:
        if analysis_type in [JointPlotAnalysisType.CORR, JointPlotAnalysisType.CROSS_CORR]:
            corr_coeff = plot_data["covariate_stats"][variables]["Correlation Coefficient"]
            p_value = plot_data["covariate_stats"][variables]["P-Value"]
            if abs(corr_coeff) >= strong_correlation and p_value <= 0.05:
                strong_correlations[variables] = corr_coeff, p_value
            elif abs(corr_coeff) >= medium_correlation and p_value <= 0.05:
                medium_correlations[variables] = corr_coeff, p_value
            elif abs(corr_coeff) >= weak_correlation and p_value <= 0.05:
                weak_correlations[variables] = corr_coeff, p_value

    return strong_correlations, medium_correlations, weak_correlations


def analyse_causal_strength(plot_data: dict) -> dict:
    """
    Analyse the strength of causal relationships in the plot data.

    Parameters
    ----------
    plot_data : dict
        Dictionary containing the data for plotting.

    Returns
    -------
    dict
        A dictionary containing strong causal relationships.
    """
    causal_variables = {}
    for variables in plot_data["covariate_stats"]:
        p_value = plot_data["covariate_stats"][variables]["P-Value"]
        if p_value <= 0.05:
            causal_variables[variables] = p_value

    return causal_variables


def text_covariate_analysis(plot_data: dict, analysis_type: JointPlotAnalysisType) -> html.Div:  # noqa: C901
    """
    Create a textual summary of the covariate analysis.

    Parameters
    ----------
    plot_data : dict
        Dictionary containing the data for plotting.
    analysis_type : JointPlotAnalysisType
        The type of correlation analysis performed.

    Returns
    -------
    html.Div
        A div containing the textual summary of the covariate analysis.
    """
    if analysis_type == JointPlotAnalysisType.CORR or analysis_type == JointPlotAnalysisType.CROSS_CORR:
        strong_correlation, medium_correlation, weak_correlation = analyse_correlation_strength(
            plot_data, analysis_type
        )
        strong_corr_text_ul = []
        medium_corr_text_ul = []
        weak_corr_text_ul = []

        if len(strong_correlation) > 0:
            strong_corr_text = html.P(
                " The following strong correlations have been observed: ",
                className="insights_calving_summary_text_no_just",
            )
            for corr_vars in strong_correlation:
                strong_corr_text_ul.append(
                    html.P(
                        f"{corr_vars.split('/')[0].split(':')[1]} and {corr_vars.split('/')[1].split(':')[1]} "
                        f"(correlation coefficient = {strong_correlation[corr_vars][0]: .2f}, "
                        f"p-value = {strong_correlation[corr_vars][1]: .3f})",
                        className="cov_analysis_summary_text_no_just_indent",
                    )
                )
        else:
            strong_corr_text = html.P("", className="cov_analysis_summary_text_no_just_indent")

        if len(medium_correlation) > 0:
            medium_corr_text = html.P(
                " The following medium correlations have been observed: ",
                className="cov_analysis_summary_text_no_just",
            )
            for corr_vars in medium_correlation:
                medium_corr_text_ul.append(
                    html.P(
                        f"{corr_vars.split('/')[0].split(':')[1]} and {corr_vars.split('/')[1].split(':')[1]} "
                        f"(correlation coefficient = {medium_correlation[corr_vars][0]: .2f}, "
                        f"p-value = {medium_correlation[corr_vars][1]: .3f})",
                        className="cov_analysis_summary_text_no_just_indent",
                    )
                )
        else:
            medium_corr_text = html.P("", className="insights_calving_summary_text_no_just_indent")

        if len(weak_correlation) > 0:
            weak_corr_text = html.P(
                " The following weak correlations have been observed: ",
                className="cov_analysis_summary_text_no_just",
            )
            for corr_vars in weak_correlation:
                weak_corr_text_ul.append(
                    html.P(
                        f"{corr_vars.split('/')[0].split(':')[1]} and {corr_vars.split('/')[1].split(':')[1]} "
                        f"(correlation coefficient = {weak_correlation[corr_vars][0]: .2f}, "
                        f"p-value = {weak_correlation[corr_vars][1]: .3f})",
                        className="cov_analysis_summary_text_no_just_indent",
                    )
                )
        else:
            weak_corr_text = html.P("", className="cov_analysis_summary_text_no_just_indent")

        text = [
            strong_corr_text,
            *strong_corr_text_ul,
            medium_corr_text,
            *medium_corr_text_ul,
            weak_corr_text,
            *weak_corr_text_ul,
        ]
    elif analysis_type == JointPlotAnalysisType.CAUSAL:
        causal_variables = analyse_causal_strength(plot_data)

        if len(causal_variables) > 0:
            causal_text = html.P(
                " The following causal relationships have been observed: ",
                className="cov_analysis_summary_text_no_just",
            )
            causal_text_ul = []
            for caus_vars in causal_variables:
                causal_text_ul.append(
                    html.P(
                        f"{caus_vars.split('/')[0].split(':')[1]} -> {caus_vars.split('/')[1].split(':')[1]} "
                        f"(p-value = {causal_variables[caus_vars]: .3f})",
                        className="cov_analysis_summary_text_no_just_indent",
                    )
                )
        else:
            causal_text = html.P("", className="cov_analysis_summary_text_no_just_indent")
            causal_text_ul = []
        text = [causal_text, *causal_text_ul]
    return html.Div(text)


def customwrap(s: str, width: int = 16, separator: str = "<br>") -> str:
    """
    Wrap text to fit within a specified width.

    Parameters
    ----------
    s : str
        The string to wrap.
    width : int, optional
        The maximum line width, by default 16
    separator : str, optional
        The separator to use for line breaks, by default "<br>"

    Returns
    -------
    str
        The wrapped string.
    """
    return separator.join(textwrap.wrap(s, width=width))


def plot_covariate_analysis(plot_data: dict, analysis_type: JointPlotAnalysisType) -> go.Figure:
    """
    Plot the results of a covariate analysis.

    Parameters
    ----------
    plot_data : dict
        Dictionary containing the data for plotting.
    analysis_type : JointPlotAnalysisType
        The type of correlation analysis performed.

    Returns
    -------
    go.Figure
        A Plotly figure containing the covariate analysis results.
    """
    colors = qualitative.Plotly
    labels = list(plot_data["variables"].keys())
    datas = [plot_data["variables"][label] for label in labels]

    # The labels have form "dataset name: variable name [units]".
    # Separate the units to a separate list for later reattachment.
    split = list(zip(*[label.rsplit(" [", 1) for label in labels], strict=True))
    titles = split[0]
    units = ["[" + x for x in split[1]]

    # Remove the dataset name, unless it would cause duplicate labels.
    labels_without_prefix = [title.split(": ", 1)[1] for title in titles]
    titles = labels_without_prefix if len(set(labels_without_prefix)) == len(labels) else labels

    # Wrap titles to a fixed width, and then reattach units on a newline.
    titles = [customwrap(title, width=14) + "<br>" + unit for title, unit in zip(titles, units, strict=True)]

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
            if analysis_type == JointPlotAnalysisType.CAUSAL:
                stats = plot_data["covariate_stats"].get(
                    f"{a_label}/{b_label}",
                    {},
                )
            else:
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
                    [f"<b>{customwrap(k)}:</b> {round(v, 2) if isinstance(v, float) else v}" for k, v in stats.items()]
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
        height=800,
        width=800,
        showlegend=False,
    )
    return fig


def plot_dataset_summary(plot_data: dict, geom: MultiPolygon) -> go.Figure:
    """Plot the output of the dataset summary module.

    Parameters
    ----------
    plot_data : dict
        The dataset summary data to plot. See `dataset_summary_module.py` for details.
    geom : MultiPolygon
        The geometry to overlay on the plot, showing the dataset query extent.

    Returns
    -------
    go.Figure
        A Plotly figure containing the dataset summary plot.
    """
    fig = go.Figure()

    if "time" in plot_data["dims"] and (
        "x" not in plot_data["dims"]
        or "y" not in plot_data["dims"]
        or (len(plot_data["dims"]["x"]["values"]) < 2 and len(plot_data["dims"]["y"]["values"]) < 2)
    ):
        # 1D timeseries
        for name, var in plot_data["vars"].items():
            values = np.array(var["values"])
            if "time" not in var["coords"]:
                continue
            ti = var["coords"].index("time")
            mean_vals = np.nanmean(
                values,
                axis=tuple(i for i in range(values.ndim) if i != ti),
                keepdims=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data["dims"]["time"]["values"],
                    y=mean_vals,
                    mode="lines",
                    name=name,
                    showlegend=True,
                )
            )
        fig.update_layout(
            xaxis={"title": "Time"},
            yaxis={"title": "Mean Value"},
            width=800,
            height=400,
        )
        return fig

    trace_i = 0
    timeless_trace_ids = set()
    trace_ids_by_time = defaultdict(list)

    wsg84 = pyproj.CRS("epsg:4326")
    crs = pyproj.CRS("epsg:3031")
    project = pyproj.Transformer.from_crs(wsg84, crs, always_xy=True).transform
    geom = transform(project, geom)
    for i, g in enumerate(geom.geoms):
        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in g.exterior.coords],
                y=[point[1] for point in g.exterior.coords],
                line=go.Line(width=1, color="black"),
                mode="lines",
                name="Query Extent",
                showlegend=(i == 0),
                legendgroup="Query Extent",
            )
        )
        timeless_trace_ids.add(trace_i)
        trace_i += 1

    xs = plot_data["dims"].get("x", {}).get("values", [])
    ys = plot_data["dims"].get("y", {}).get("values", [])

    for name, var in plot_data["vars"].items():
        values = np.array(var["values"])
        if "x" not in var["coords"] or "y" not in var["coords"]:
            continue
        xi = var["coords"].index("x")
        yi = var["coords"].index("y")

        if "time" not in var["coords"]:
            mean_vals = np.nanmean(
                values,
                axis=tuple(i for i in range(values.ndim) if i not in (xi, yi)),
                keepdims=False,
            )
            fig.add_trace(
                go.Heatmap(
                    x=xs,
                    y=ys,
                    z=mean_vals,
                    name=name,
                    showlegend=True,
                )
            )
            timeless_trace_ids.add(trace_i)
            trace_i += 1
            continue

        ti = var["coords"].index("time")
        removed_axes = tuple(i for i in range(values.ndim) if i not in (xi, yi, ti))
        mean_vals = np.nanmean(
            values,
            axis=removed_axes,
            keepdims=False,
        )
        valmin, valmax = np.nanmin(mean_vals), np.nanmax(mean_vals)

        # Need to update ti after mean calculation
        ti -= len([ra for ra in removed_axes if ra < ti])

        for t in range(values.shape[ti]):
            fig.add_trace(
                go.Heatmap(
                    x=xs,
                    y=ys,
                    z=np.take(mean_vals, t, axis=ti),
                    zmin=valmin,
                    zmax=valmax,
                    name=name,
                    visible=(t == 0),
                    legendgroup=name,
                )
            )
            trace_ids_by_time[t].append(trace_i)
            trace_i += 1

    if len(trace_ids_by_time) > 1:
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "Time: "},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "label": datetime.fromtimestamp(t).strftime("%Y-%m-%d"),
                            "method": "update",
                            "args": [
                                {
                                    "visible": [
                                        (trace_id in trace_ids_by_time[i]) or (trace_id in timeless_trace_ids)
                                        for trace_id in range(trace_i)
                                    ]
                                },
                            ],
                        }
                        for i, t in enumerate(plot_data["dims"]["time"]["values"])
                    ],
                }
            ],
            xaxis={"title": "X Coordinate"},
            yaxis={"title": "Y Coordinate"},
            width=800,
            height=800,
        )
    return fig
