from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import mapping
import numpy as np
import plotting
import process_data
import query_data
from dash import Dash, Input, Output, State, callback, html

ice_shelf_inital = "thwaites"

# Import data ###
# ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj = query_data.query_ice_shelf_wide_data()
# ocean_polygons_epsg4326, ocean_polygons_proj = query_data.query_all_ocean_data()
ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj, ocean_polygons_epsg4326, ocean_polygons_proj = (
    query_data.query_polygons()
)
ice_sheet_polygons_epsg4326, ice_sheet_polygons_proj = query_data.query_ice_sheet_data()
summary_data_initial, timeseries_container, _, _ = query_data.query_summary_data(
    ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf_inital
)


# Build Dash app ###
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

app.layout = html.Div(
    [
        dash.page_container,
    ]
)


@callback(Output("ice_shelf_map", "children"), Input("ice_shelf_selection", "value"))
def update_graph(value):
    ice_shelf_map = mapping.create_ice_shelf_map(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ocean_polygons=ocean_polygons_proj,
        ice_sheet_polygons=ice_sheet_polygons_proj,
        ice_shelf=value,
    )

    return ice_shelf_map


@callback(
    Output("iceshelf-summary-text-div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def update_ice_shelf_summary(ice_shelf, year_range):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    total_mass_loss, negative_cummulative_contributors, derivative_summary = process_data.generate_statistics(
        timeseries_container, year_range
    )
    year_diff = year_range[1] - year_range[0]

    neg_list = []
    for var in negative_cummulative_contributors.index:
        neg_list.append(
            html.Li(
                f"{' '.join(var.split('_')).capitalize()} has contributed approximately {int(np.round(np.abs(negative_cummulative_contributors.loc[var, 'Total'])))} Gt to this total loss over this period."
            )
        )
    return html.Div(
        [
            html.P(" "),
            html.H5(f"{ice_shelf} Ice Shelf Summary:"),
            html.P(
                f"Overall the {ice_shelf} Ice Shelf has lost a total mass of approximately {int(np.round(np.abs(total_mass_loss)))} Gt over the past {year_diff} years. There are a number of key contributing factors to this ice loss:",
                style={"padding-right": "100px", "padding-top": "10px"},
            ),
            html.Ul(neg_list),
            html.P(
                f"Other important sources of ice loss from {ice_shelf} ice shelf in the {year_diff} year period are:",
                style={"padding-right": "100px", "padding-top": "0px"},
            ),
            html.Ul(
                [
                    html.Li(
                        f"There has been approximately {int(np.round(np.abs(derivative_summary.loc['calving', 'Total'])))} Gt of ice lost from the ice shelf as a result of calving."
                    ),
                    html.Li(
                        f"There has been approximately {int(np.round(np.abs(derivative_summary.loc['discharge', 'Total'])))} Gt of ice discharged as water."
                    ),
                    html.Li(
                        f"Finally, there has been approximately {int(np.round(np.abs(derivative_summary.loc['basal_melt_flux', 'Total'])))} Gt of ice lost through basal melting."
                    ),
                ]
            ),
            html.H5(
                "Timeseries Analysis:",
                style={"padding-right": "100px", "padding-top": "10px"},
            ),
            html.P("Use the toggle switches below to investigate temporal trends in key Ice shelf parameters."),
        ]
    )


@callback(
    Output("figure_1_div", "style"),
    Input("switch-cumulative-mass-change", "value"),
)
def plot_summary_indicator1_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_4_div", "style"),
    Input("switch-cumulative-thinning", "value"),
)
def plot_summary_indicator2_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_2_div", "style"),
    Input("switch-cumulative-discharge-change", "value"),
)
def plot_summary_indicator3_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_5_div", "style"),
    Input("switch-cumulative-SBM-anomolies", "value"),
)
def plot_summary_indicator4_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_3_div", "style"),
    Input("switch-cumulative-retreat", "value"),
)
def plot_summary_indicator5_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_6_div", "style"),
    Input("switch-mass-balance", "value"),
)
def plot_summary_indicator6_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_7_div", "style"),
    Input("switch-calving-rate", "value"),
)
def plot_summary_indicator7_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_9_div", "style"),
    Input("switch-basal-melt-flux", "value"),
)
def plot_summary_indicator8_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_8_div", "style"),
    Input("switch-smb", "value"),
)
def plot_summary_indicator9_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_10_div", "style"),
    Input("switch-discharge-rate", "value"),
)
def plot_summary_indicator10_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("indicator_1_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_1_store", "data"),
)
def plot_summary_indicator1(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_2_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_2_store", "data"),
)
def plot_summary_indicator2(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_3_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_3_store", "data"),
)
def plot_summary_indicator3(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_4_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_4_store", "data"),
)
def plot_summary_indicator4(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_5_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_5_store", "data"),
)
def plot_summary_indicator5(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_indicator(summary_data, variable, year_range)
    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_1_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_1_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_2_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_2_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_3_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_3_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_4_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_4_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("indicator_5_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_5_store_deriv", "data"),
)
def plot_summary_timeseries1(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    # fig = plotting.plot_summary_timeseries(summary_data, variable, year_range)
    fig = plotting.plot_summary_timeseries(timeseries_container.timeseries[variable])
    return fig


@callback(
    Output("gauge_1_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_1_store", "data"),
)
def plot_gauge_indicator1(ice_shelf, variable):
    summary_data, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf
    )

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("gauge_2_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_2_store", "data"),
)
def plot_gauge_indicator2(ice_shelf, variable):
    summary_data, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf
    )

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("gauge_3_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_3_store", "data"),
)
def plot_gauge_indicator3(ice_shelf, variable):
    summary_data, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf
    )

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("gauge_4_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_4_store", "data"),
)
def plot_gauge_indicator4(ice_shelf, variable):
    summary_data, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf
    )

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("gauge_5_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    State("gauge_5_store", "data"),
)
def plot_gauge_indicator5(ice_shelf, variable):
    summary_data, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf
    )

    fig = plotting.plot_gauge_indicator(summary_data, variable)
    return fig


@callback(
    Output("results-plot", "children"),
    Input("run_analysis-button", "n_clicks"),
    State("ice_shelf_selection", "value"),
    State("summary_year_selector", "value"),
    State("select-analysis-type", "value"),
    State("thwaites-paper-moorings", "value"),
    State("thwaites-paper-basal-melt-anoms-pi", "value"),
    State("thwaites-paper-basal-melt-anoms-thwaites", "value"),
    State("thwaites-paper-subglacial-lake-disch", "value"),
)
def conduct_covariate_analysis(
    n,
    ice_shelf,
    year_range,
    analysis_type,
    thwaites_paper_moorings,
    thwaites_paper_bm_anoms_pi,
    thwaites_paper_bm_anoms_thw,
    thwaites_paper_subglacial_lake_disch,
):
    ice_shelf_extent = ice_shelf_polygons_epsg4326[ice_shelf_polygons_epsg4326["Name"] == ice_shelf]["geometry"]
    ocean_extent = ocean_polygons_epsg4326[ocean_polygons_epsg4326["Name"] == ice_shelf]["geometry"]

    timeseries = []
    if thwaites_paper_moorings:
        timeseries.append(
            query_data.query_local_data(
                dataset_id="MOORINGS_1D",
                variable="",
                extent_polygon_epsg4326=ice_shelf_extent.values[0],
                start_time=datetime(year_range[0], 1, 1),
                end_time=datetime(year_range[1], 1, 1),
            )
        )


if __name__ == "__main__":
    app.run(debug=True)
