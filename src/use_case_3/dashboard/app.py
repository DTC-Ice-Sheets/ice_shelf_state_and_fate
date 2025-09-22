import dash
import dash_bootstrap_components as dbc
import mapping
import plotting
import query_data
from dash import Dash, Input, Output, State, callback, html

ice_shelf_inital = "Thwaites"

# Import data ###
ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj = query_data.query_ice_shelf_wide_data()
summary_data_initial, _, _ = query_data.query_summary_data(
    ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf_inital
)

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
    Output("iceshelf-summary-text-div", "value"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def update_ice_shelf_summary(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


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
    Output("figure_2_div", "style"),
    Input("switch-cumulative-thinning", "value"),
)
def plot_summary_indicator2_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("figure_3_div", "style"),
    Input("switch-cumulative-discharge-change", "value"),
)
def plot_summary_indicator3_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("figure_4_div", "style"),
    Input("switch-cumulative-SBM-anomolies", "value"),
)
def plot_summary_indicator4_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("figure_5_div", "style"),
    Input("switch-cumulative-retreat", "value"),
)
def plot_summary_indicator5_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("figure_6_div", "style"),
    Input("switch-mass-balance", "value"),
)
def plot_summary_indicator6_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("figure_7_div", "style"),
    Input("switch-calving-rate", "value"),
)
def plot_summary_indicator7_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("figure_8_div", "style"),
    Input("switch-basal-melt-flux", "value"),
)
def plot_summary_indicator8_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("figure_9_div", "style"),
    Input("switch-smb", "value"),
)
def plot_summary_indicator9_visable(visability):
    if visability:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("figure_10_div", "style"),
    Input("switch-discharge-rate", "value"),
)
def plot_summary_indicator10_visable(visability):
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
