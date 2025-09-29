import base64
from io import BytesIO

import dash
import dash_bootstrap_components as dbc
import mapping
import numpy as np
import geopandas as gpd
import plotting
import process_data
import query_data
import yaml
from dash import Dash, Input, Output, State, callback, html, ALL, MATCH

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    JointPlotAnalysisType,
)

with open("/home/palmerjohn/code/dtc_ice_sheets/use_case_3/src/use_case_3/dashboard/variable_mappings.yml") as f:
    config_dict = yaml.safe_load(f)

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


@callback(
    Output("insights_mass_balance_div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_massbalance(ice_shelf, year_range):
    if ice_shelf is not None:
        _, timeseries_container, _, _ = query_data.query_summary_data(
            ice_shelf_polygons=ice_shelf_polygons_proj,
            ice_shelf=ice_shelf,
            year_range=year_range,
        )
        process_data.calculate_mass_balance(timeseries_container=timeseries_container)
    else:
        return html.Div()


@callback(
    Output("insights_calving_summary_div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_calving(ice_shelf, year_range):
    if ice_shelf is not None:
        _, timeseries_container, _, _ = query_data.query_summary_data(
            ice_shelf_polygons=ice_shelf_polygons_proj,
            ice_shelf=ice_shelf,
            year_range=year_range,
        )

        total, total_unc, calving_events, calving_unc_events, time_events = process_data.detect_calving_events(
            timeseries_container=timeseries_container
        )
        if len(calving_events) == 0:
            return html.Div()
        else:
            title = html.H6("Calving Events Detected:", className="insights_h6")
            if len(calving_events) == 1:
                main = html.P(
                    f"There was one significant calving event on {ice_shelf.replace('_', ' ').title()} Ice Shelf between {year_range[0]} and {year_range[1]}. This calving event occured in {time_events[0]}, and resulted an overall ice loss of {calving_events[0]: .0f}+/-{calving_unc_events[0]: .0f}Gt.",
                    className="insights_calving_summary_text",
                )
            else:
                main = html.P(
                    f"There were {len(calving_events)} significant calving events on {ice_shelf.replace('_', ' ').title()} Ice Shelf between {year_range[0]} and {year_range[1]}. These calving events have resulted in an overal ice loss of approximately {total: .0f}+/-{total_unc: .0f}Gt. The largest calving event was {calving_events[0]: .0f}+/-{calving_unc_events[0]: .0f}Gt and occured in {time_events[0]}.",
                    className="insights_calving_summary_text",
                )

            text = [
                title,
                main,
            ]
            return html.Div(text)
    else:
        return html.Div()


# These need to be refactored to be a pattern matching callback so there is only one for all of the switches.
@callback(
    Output(f"switch-group-vars-{'polarice_elevation_change'.lower()}", "style"),
    Input(f"ds_{'polarice_elevation_change'.lower()}", "value"),
)
def elev_polarice_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


# @callback(
#     Output(f"switch-group-vars-{'elevation_change_eolis'.lower()}", "style"),
#     Input(f"ds_{'elevation_change_eolis'.lower()}", "value"),
# )
# def elev_eolis_visable(visability):
#     if visability:
#         return {"display": "block"}
#     return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'rema_v2'.lower()}", "style"),
    Input(f"ds_{'rema_v2'.lower()}", "value"),
)
def rema_v2_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'bedmachine_v3'.lower()}", "style"),
    Input(f"ds_{'bedmachine_v3'.lower()}", "value"),
)
def bedmachine_v3_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'polar_plus_ice_shelves_basal_melt'.lower()}", "style"),
    Input(f"ds_{'polar_plus_ice_shelves_basal_melt'.lower()}", "value"),
)
def polar_ice_plus_bm_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'polar_plus_ice_shelves_calving_front'.lower()}", "style"),
    Input(f"ds_{'polar_plus_ice_shelves_calving_front'.lower()}", "value"),
)
def polar_ice_plus_cf_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'sub_glacial_lakes'.lower()}", "style"),
    Input(f"ds_{'sub_glacial_lakes'.lower()}", "value"),
)
def subglacial_lakes_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'its_live_basal_melt'.lower()}", "style"),
    Input(f"ds_{'its_live_basal_melt'.lower()}", "value"),
)
def its_live_bm_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'its_live_velocity'.lower()}", "style"),
    Input(f"ds_{'its_live_velocity'.lower()}", "value"),
)
def its_live_vel_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'racmo'.lower()}", "style"),
    Input(f"ds_{'racmo'.lower()}", "value"),
)
def racmo_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'ocean_reanalysis'.lower()}", "style"),
    Input(f"ds_{'ocean_reanalysis'.lower()}", "value"),
)
def ocean_reanalysis_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'sea_ice_concentration'.lower()}", "style"),
    Input(f"ds_{'sea_ice_concentration'.lower()}", "value"),
)
def sea_ice_concentration_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'sea_ice_edge'.lower()}", "style"),
    Input(f"ds_{'sea_ice_edge'.lower()}", "value"),
)
def sea_ice_edge_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'sea_ice_thickness'.lower()}", "style"),
    Input(f"ds_{'sea_ice_thickness'.lower()}", "value"),
)
def sea_ice_thickness_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'ocean_moorings'.lower()}", "style"),
    Input(f"ds_{'ocean_moorings'.lower()}", "value"),
)
def ocean_moorings_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'southern_annular_mode'.lower()}", "style"),
    Input(f"ds_{'southern_annular_mode'.lower()}", "value"),
)
def souther_annular_mode_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'basal_melt_anomalies_pine_island_1d'.lower()}", "style"),
    Input(f"ds_{'basal_melt_anomalies_pine_island_1d'.lower()}", "value"),
)
def thwaites_bm_anoms_pi_1d_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'basal_melt_anomalies_thwaites_1d'.lower()}", "style"),
    Input(f"ds_{'basal_melt_anomalies_thwaites_1d'.lower()}", "value"),
)
def thwaites_bm_anoms_thw_1d_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


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
            html.H5(f"{' '.join(ice_shelf.split('_')).title()} Ice Shelf Summary:"),
            html.P(
                f"Overall the {' '.join(ice_shelf.split('_')).title()} Ice Shelf has lost a total mass of approximately {int(np.round(np.abs(total_mass_loss)))} Gt over the specified {year_diff} year period. There are a number of key contributing factors to this ice loss:",
                style={"padding-right": "100px", "padding-top": "10px"},
            ),
            html.Ul(neg_list),
            html.P(
                f"Other important sources of ice loss from {' '.join(ice_shelf.split('_')).title()} ice shelf in the {year_diff} year period are:",
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
            html.P("Use the toggle switches below to investigate temporal trends in key ice shelf parameters."),
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
    State({"type": "data_selector", "index": ALL}, "id"),
    State({"type": "data_selector", "index": ALL}, "value"),
    # State("moorings_1d-depth_PIG_N_and_S_combined", "value"),
    # State("basal_melt_anomalies_pine_island_1d-basal_melt_anomalies", "value"),
    # State("subglacial_lake_discharge_1d-lake_discharge", "value"),
    # # State("subglacial_lakes_volume_change_1d-volume_change", "value"),
    # State("basal_melt_anomalies_thwaites_1d-basal_melt_anomalies", "value"),
)
def conduct_covariate_analysis(
    n,
    ice_shelf,
    year_range,
    analysis_type,
    dataset_ids,
    datasets,
):
    ice_shelf_extent = ice_shelf_polygons_epsg4326[ice_shelf_polygons_epsg4326["Name"] == ice_shelf]["geometry"].values[
        0
    ]
    ocean_extent = ocean_polygons_epsg4326[ocean_polygons_epsg4326["Name"] == ice_shelf]["geometry"].values[0]
    climate_extent = None

    # polygons = {"ice_shelf": ice_shelf_extent, "ocean": ocean_extent, "climate": climate_extent}

    polygons = gpd.GeoDataFrame(
        {"Name": ["ice_shelf", "ocean", "climate"], "geometry": [ice_shelf_extent, ocean_extent, climate_extent]}
    )
    polygons = polygons.set_crs("epsg: 4326").set_index("Name")

    variables = [dataset_ids[i]["index"] for i, dataset in enumerate(datasets) if dataset]
    # variab = [dataset_ids[i]["index"] for i, dataset in enumerate(datasets) if dataset]
    # variables = [
    #     "moorings_1d-depth_PIG_N_and_S_combined",
    #     "basal_melt_anomalies_pine_island_1d-basal_melt_anomalies",
    #     "subglacial_lakes_volume_change_1d-volume_change",
    #     "basal_melt_anomalies_thwaites_1d-basal_melt_anomalies",
    # ]

    axis_labels = []
    for variable in variables:
        # print(variable)
        axis_labels.append(
            config_dict[variable.split("-")[0].lower()]["variables"][variable.split("-")[-1]]["y_axis_label"]
        )

    timeseries = query_data.process_all_data(variables, polygons, year_range)

    if len(timeseries) > 0:
        timeseries_container = TimeSeriesPreprocessor(
            timeseries=timeseries,
            time_period=AggregationPeriod.MONTH,
            aggregation=AggregationStrategy.MEAN,
        )

        if analysis_type == "1":
            analysis_type_enum = JointPlotAnalysisType.CORR
        elif analysis_type == "2":
            analysis_type_enum = JointPlotAnalysisType.CROSS_CORR
        elif analysis_type == "3":
            analysis_type_enum = JointPlotAnalysisType.CAUSAL
        covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
        fig, axes = covariate_analyser.covariate_analysis(
            max_lag=3, axis_labels=axis_labels, analysis_type=analysis_type_enum
        )
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        image_string = base64.b64encode(buffer.getvalue()).decode()

        results_plot = "data:image/png;base64," + image_string
        results_plot = html.Img(src=results_plot, style={"max-width": "100%", "max-height": "100%"})

        return results_plot
    return None


if __name__ == "__main__":
    app.run(debug=True)
