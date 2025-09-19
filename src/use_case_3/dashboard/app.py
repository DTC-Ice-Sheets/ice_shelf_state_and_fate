import dash
import dash_bootstrap_components as dbc
import mapping
import plotting
import query_data
from dash import Dash, Input, Output, State, callback, html

# from use_case_3.dashboard.dashboard_layout import column_summary_guages, column_summary_map
# from use_case_3.dashboard.dashboard_callbacks import *

ice_shelf_inital = "Thwaites"

# Import data ###
ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj = query_data.query_ice_shelf_wide_data()
summary_data_initial, _, _ = query_data.query_summary_data(
    ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf_inital
)

# print(summary_data[variable_initial])

# Create the initial map ###
ice_shelf_map = mapping.create_ice_shelf_map(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf_inital)

# Build Dash app ###
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

app.layout = html.Div(
    [
        dash.page_container,
    ]
)


@callback(Output("ice_shelf_map", "children"), Input("ice_shelf_selection", "value"))
def update_graph(value):
    ice_shelf_map = mapping.create_ice_shelf_map(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=value)

    return ice_shelf_map


@callback(
    Output("figure_1_div", "style"),
    Input("switch-cumulative-mass-change", "value"),
)
def plot_summary_indicator1_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("indicator_1_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_1_store", "data"),
)
def plot_summary_indicator1(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_2_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_2_store", "data"),
)
def plot_summary_indicator2(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_3_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_3_store", "data"),
)
def plot_summary_indicator3(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_4_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_4_store", "data"),
)
def plot_summary_indicator4(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_5_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_5_store", "data"),
)
def plot_summary_indicator5(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_1_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_1_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_2_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_2_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_3_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_3_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_4_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_4_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("indicator_5_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_5_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    return fig


@callback(
    Output("gauge_1_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_1_store", "data"),
)
def plot_gauge_indicator1(ice_shelf, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("gauge_2_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_2_store", "data"),
)
def plot_gauge_indicator2(ice_shelf, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("gauge_3_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_3_store", "data"),
)
def plot_gauge_indicator3(ice_shelf, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("gauge_4_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_4_store", "data"),
)
def plot_gauge_indicator4(ice_shelf, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("gauge_5_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_5_store", "data"),
)
def plot_gauge_indicator5(ice_shelf, variable):
    summary_data, _, _ = query_data.query_summary_data(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf)

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
