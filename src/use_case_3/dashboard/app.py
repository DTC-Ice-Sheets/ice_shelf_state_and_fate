import base64
import logging
import os
from io import BytesIO
from time import sleep

# from asyncio import sleep
import boto3
import dash
import dash_bootstrap_components as dbc
import diskcache
import dotenv
import geopandas as gpd
import mapping
import numpy as np
import pandas as pd
import plotting
import process_data
import query_data
import requests
import yaml
from dash import ALL, Dash, DiskcacheManager, Input, Output, State, callback, dcc, html, set_props
from num2words import num2words

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    JointPlotAnalysisType,
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

url = "http://localhost:8000"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    endpoint_url=os.environ["AWS_S3_ENDPOINT"],
    region_name=os.environ["AWS_DEFAULT_REGION"],
    use_ssl=True,
)

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

with open("/home/palmerjohn/code/dtc_ice_sheets/use_case_3/src/use_case_3/dashboard/variable_mappings.yml") as f:
    config_dict = yaml.safe_load(f)

ice_shelf_inital = "thwaites"

# Import data ###
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
    Output("insights_headline_title_mass_balance_div", "children"),
    Output("insights_headline_text_mass_balance_div", "children"),
    Output("insights_btn_show_mass_balance_div", "style"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_massbalance_headline(ice_shelf, year_range):
    if ice_shelf is not None:
        _, timeseries_container, _, _ = query_data.query_summary_data(
            ice_shelf_polygons=ice_shelf_polygons_proj,
            ice_shelf=ice_shelf,
            year_range=year_range,
        )

        mass_balance_events, mass_balance_rate, calving_matched_years = process_data.calculate_mass_balance(
            timeseries_container=timeseries_container
        )

        if mass_balance_events[0] < 0:
            gained_lost = "lost"
        else:
            gained_lost = "gained"
        return (
            html.Div(html.P(html.Strong("MASS BALANCE:"), className="insights_headline_text")),
            html.Div(
                html.P(
                    f"In the past {year_range[1] - year_range[0]} years, {ice_shelf.replace('_', ' ').title()} Ice Shelf has {gained_lost} {np.abs(mass_balance_events[0]): .0f} Gt of ice.",
                    className="insights_headline_text",
                )
            ),
            {
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
            },
        )
    return html.Div(), html.Div(), {"display": "none"}


@callback(
    Output("insights_mass_balance_div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_massbalance_detail(ice_shelf, year_range):
    if ice_shelf is not None:
        _, timeseries_container, _, _ = query_data.query_summary_data(
            ice_shelf_polygons=ice_shelf_polygons_proj,
            ice_shelf=ice_shelf,
            year_range=year_range,
        )
        mass_balance_events, mass_balance_rate, calving_matched_years = process_data.calculate_mass_balance(
            timeseries_container=timeseries_container
        )

        if mass_balance_events[0] < 0:
            gained_lost = "lost"
        else:
            gained_lost = "gained"

        if mass_balance_rate < 0:
            increasing_decreasing = "increasing"
        else:
            increasing_decreasing = "decreasing"

        if len(calving_matched_years) > 0:
            calving_matched_text = (
                f" It is likely that {num2words(len(calving_matched_years))} of these events are due to calving."
            )
        else:
            calving_matched_text = " Currently there are no obvious causes of these events"
        # title = html.H6("Mass Balance:", className="insights_h6")
        # intro = html.P(
        intro = dcc.Markdown(
            f"In the past {year_range[1] - year_range[0]} years, {ice_shelf.replace('_', ' ').title()} Ice Shelf has {gained_lost} {np.abs(mass_balance_events[0]): .0f}+/-{mass_balance_events[1]: .0f}Gt of ice. This ice loss is currently {increasing_decreasing} at a rate of {np.abs(mass_balance_rate): .1f} Gt/yr <sup>2</sup>.",
            dangerously_allow_html=True,
            className="insights_calving_summary_text",
        )
        main = html.P()
        if len(mass_balance_events[2]) > 0:
            if len(mass_balance_events[2]) == 1:
                main = html.P(
                    f"There was one significant event on {ice_shelf.replace('_', ' ').title()} Ice Shelf between {year_range[0]} and {year_range[1]}. This event occured in {mass_balance_events[4][-1]}, and resulted an overall ice loss of {mass_balance_events[2][-1]: .0f}+/-{mass_balance_events[3][-1]: .0f}Gt.",
                    className="insights_calving_summary_text",
                )
            else:
                main = html.P(
                    f"There were {num2words(len(mass_balance_events[2]))} significant events impacting {ice_shelf.replace('_', ' ').title()} Ice Shelf mass balance between {year_range[0]} and {year_range[1]}."
                    + calving_matched_text,
                    className="insights_calving_summary_text",
                )

        text = [
            # title,
            intro,
            main,
        ]
        return html.Div(text)
    return html.Div()


@callback(
    Output("mass_balance_graph", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def plot_mass_balance(ice_shelf, year_range):
    if ice_shelf is None:
        ice_shelf = "thwaites"
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    mass_balance_events, _, _ = process_data.calculate_mass_balance(timeseries_container=timeseries_container)

    fig = plotting.plot_summary_timeseries(
        timeseries_container.timeseries["mass_balance"],
        outliers=(mass_balance_events[4], mass_balance_events[2]),
        margin={"t": 0, "l": 150, "b": 25, "r": 50},
        y_label="Mass Balance (Gt/year)",
    )
    return fig


@callback(
    Output("insights_mass_balance_div", "style"),
    Output("mass_balance_graph_div", "style"),
    Input("insights_btn_show_mass_balance", "n_clicks"),
)
def button_detailed_mass_balance(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}


@callback(
    Output("insights_headline_title_calving_div", "children"),
    Output("insights_headline_text_calving_div", "children"),
    Output("insights_btn_show_calving_div", "style"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_calving_headline(ice_shelf, year_range):
    if ice_shelf is not None:
        _, timeseries_container, _, _ = query_data.query_summary_data(
            ice_shelf_polygons=ice_shelf_polygons_proj,
            ice_shelf=ice_shelf,
            year_range=year_range,
        )

        _, _, calving_events, _, _ = process_data.detect_outlier_events(timeseries_container=timeseries_container)
        if len(calving_events) > 0:
            if len(calving_events) > 1:
                has_have = "have"
                event_s = "events"
                year_s = "years"
            else:
                has_have = "has"
                event_s = "event"
                year_s = "year"

            return (
                html.Div(html.P(html.Strong("CALVING:"), className="insights_headline_text")),
                html.Div(
                    html.P(
                        f"There {has_have} been {num2words(len(calving_events))} {year_s} with large calving {event_s} in the {year_range[1] - year_range[0]} year period chosen.",
                        className="insights_headline_text",
                    )
                ),
                # {"display": "block"},
                {
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    # "display": "block",
                    # "width": "100%",
                },
            )
    return html.Div(), html.Div(), {"display": "none"}


@callback(
    Output("insights_calving_detail_div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_calving_detail(ice_shelf, year_range):
    if ice_shelf is not None:
        _, timeseries_container, _, _ = query_data.query_summary_data(
            ice_shelf_polygons=ice_shelf_polygons_proj,
            ice_shelf=ice_shelf,
            year_range=year_range,
        )

        total, total_unc, calving_events, calving_unc_events, time_events = process_data.detect_outlier_events(
            timeseries_container=timeseries_container
        )
        if len(calving_events) == 0:
            return html.Div()
        if len(calving_events) == 1:
            main = html.P(
                f"There was one significant calving event on {ice_shelf.replace('_', ' ').title()} Ice Shelf between {year_range[0]} and {year_range[1]}. This calving event occured in {time_events[0]}, and resulted an overall ice loss of {calving_events[0]: .0f}+/-{calving_unc_events[0]: .0f}Gt.",
                className="insights_calving_summary_text",
            )
        else:
            main = html.P(
                f"There were {num2words(len(calving_events))} significant calving events on {ice_shelf.replace('_', ' ').title()} Ice Shelf between {year_range[0]} and {year_range[1]}. These calving events have resulted in an overal ice loss of approximately {total: .0f}+/-{total_unc: .0f}Gt. The largest calving event was {calving_events[-1]: .0f}+/-{calving_unc_events[-1]: .0f}Gt and occured in {time_events[-1]}.",
                className="insights_calving_summary_text",
            )

        text = [
            main,
        ]
        return html.Div(text)
    return html.Div()


@callback(
    Output("calving_graph", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def plot_calving(ice_shelf, year_range):
    if ice_shelf is None:
        ice_shelf = "thwaites"

    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    calving_events = process_data.detect_outlier_events(
        timeseries_container=timeseries_container, variable="calving", stationary_check=False
    )

    fig = plotting.plot_summary_timeseries(
        timeseries_container.timeseries["calving"],
        outliers=(calving_events[4], calving_events[2]),
        margin={"t": 0, "l": 150, "b": 25, "r": 50},
        y_label="Calving (Gt/year)",
    )
    return fig


@callback(
    Output("insights_calving_detail_div", "style"),
    Output("calving_graph_div", "style"),
    Input("insights_btn_show_calving", "n_clicks"),
)
def button_detailed_calving(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}


@callback(
    Output("insights_headline_title_sg_lakes_div", "children"),
    Output("insights_headline_text_sg_lakes_div", "children"),
    Output("insights_btn_show_sg_lakes_div", "style"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_sg_lake_headline(ice_shelf, year_range):
    if ice_shelf == "thwaites":
        lakes_events, _, _ = process_data.example_subglacial_lake_analysis(
            ice_shelf, ice_shelf_polygons_epsg4326, ocean_polygons_epsg4326, year_range
        )

        if len(lakes_events) > 0:
            if len(lakes_events) > 1:
                has_have = "have"
                event_s = "events"
                year_s = "years"
                are_is = "are"
            else:
                has_have = "has"
                event_s = "event"
                year_s = "year"
                are_is = "is"

            return (
                html.Div(html.P(html.Strong("SUBGLACIAL LAKES:"), className="insights_headline_text")),
                html.Div(
                    html.P(
                        f"There {are_is} {num2words(len(lakes_events))} subglacial lakes with large volume changes {event_s} in the {year_range[1] - year_range[0]} year period chosen.",
                        className="insights_headline_text",
                    )
                ),
                {
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                },
            )
    return html.Div(), html.Div(), {"display": "none"}


# @callback(
#     Output("insights_sg_lakes_detail_div", "children"),
#     Input("ice_shelf_selection", "value"),
#     Input("summary_year_selector", "value"),
# )
# def insight_update_sg_lake_detail(ice_shelf, year_range):
#     if ice_shelf == "thwaites":
#         _, timeseries_container, _, _ = query_data.query_summary_data(
#             ice_shelf_polygons=ice_shelf_polygons_proj,
#             ice_shelf=ice_shelf,
#             year_range=year_range,
#         )

#         lakes_events, results, _ = process_data.example_subglacial_lake_analysis(
#             ice_shelf, ice_shelf_polygons_epsg4326, ocean_polygons_epsg4326, year_range
#         )

#         total, total_unc, calving_events, calving_unc_events, time_events = process_data.detect_outlier_events(
#             timeseries_container=timeseries_container
#         )
#         if len(calving_events) == 0:
#             return html.Div()
#         else:
#             if len(calving_events) == 1:
#                 main = html.P(
#                     f"There was one significant calving event on {ice_shelf.replace('_', ' ').title()} Ice Shelf between {year_range[0]} and {year_range[1]}. This calving event occured in {time_events[0]}, and resulted an overall ice loss of {calving_events[0]: .0f}+/-{calving_unc_events[0]: .0f}Gt.",
#                     className="insights_calving_summary_text",
#                 )
#             else:
#                 main = html.P(
#                     f"There were {len(calving_events)} significant calving events on {ice_shelf.replace('_', ' ').title()} Ice Shelf between {year_range[0]} and {year_range[1]}. These calving events have resulted in an overal ice loss of approximately {total: .0f}+/-{total_unc: .0f}Gt. The largest calving event was {calving_events[-1]: .0f}+/-{calving_unc_events[-1]: .0f}Gt and occured in {time_events[-1]}.",
#                     className="insights_calving_summary_text",
#                 )

#             text = [
#                 main,
#             ]
#             return html.Div(text)
#     else:
#         return html.Div()


@callback(
    Output("sg_lakes_graph_vol_change", "figure"),
    Output("sg_lakes_graph_rate_vol_change", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def plot_sg_lake(ice_shelf, year_range):
    ice_shelf = "thwaites"

    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    calving_events = process_data.detect_outlier_events(
        timeseries_container=timeseries_container, variable="calving", stationary_check=False
    )

    lakes_events, results, timeseries_container = process_data.example_subglacial_lake_analysis(
        ice_shelf, ice_shelf_polygons_epsg4326, ocean_polygons_epsg4326, year_range
    )

    var = "volume_change"

    variables = [
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW70-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW124-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW142-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW170-" + var,
    ]
    fig_vol_change = plotting.plot_summary_timeseries_multiple(
        timeseries_container,
        variables=variables,
        outliers=(),
        margin={"t": 0, "l": 150, "b": 25, "r": 50},
        y_label="Volume Change (#)",
    )

    var = "rate_of_volume_change"

    variables = [
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW70-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW124-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW142-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW170-" + var,
    ]
    fig_rate_vol_change = plotting.plot_summary_timeseries_multiple(
        timeseries_container,
        variables=variables,
        outliers=(),
        margin={"t": 0, "l": 150, "b": 25, "r": 50},
        y_label="Rate of Volume Change (#)",
    )

    return fig_vol_change, fig_rate_vol_change


@callback(
    Output("insights_sg_lakes_detail_div", "style"),
    Output("sg_lakes_graph_vol_change_div", "style"),
    Output("sg_lakes_graph_rate_vol_change_div", "style"),
    Input("insights_btn_show_sg_lakes", "n_clicks"),
)
def button_detailed_sg_lake(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}, {"display": "none"}


@callback(
    Output("insights_headline_title_basal_div", "children"),
    Output("insights_headline_text_basal_div", "children"),
    Output("insights_btn_show_basal_div", "style"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_basal_headline(ice_shelf, year_range):
    if ice_shelf is not None:
        _, timeseries_container, _, _ = query_data.query_summary_data(
            ice_shelf_polygons=ice_shelf_polygons_proj,
            ice_shelf=ice_shelf,
            year_range=year_range,
        )

        bm_change_brackets = process_data.dectect_changes(dataset=timeseries_container, variable="basal_melt_flux")

        if len(bm_change_brackets) > 0:
            medians = process_data.calculate_median_change_brackets(
                dataset=timeseries_container,
                variable="basal_melt_flux",
                change_brackets=bm_change_brackets,
            )

            median_change = (medians[0] - medians[-1]) / medians[0] * 100

            mag = "decrease" if median_change >= 0 else "increase"

            return (
                html.Div(html.P(html.Strong("BASAL MELT FLUX:"), className="insights_headline_text")),
                html.Div(
                    html.P(
                        f"There has been a {np.abs(median_change): .1f}% {mag} in basal melt flux in the selected {year_range[1] - year_range[0]} year period.",
                        className="insights_headline_text",
                    )
                ),
                {
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                },
            )
    return html.Div(), html.Div(), {"display": "none"}


@callback(
    Output("insights_basal_detail_div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_basal_melt_flux_detail(ice_shelf, year_range):
    if ice_shelf is not None:
        _, timeseries_container, _, _ = query_data.query_summary_data(
            ice_shelf_polygons=ice_shelf_polygons_proj,
            ice_shelf=ice_shelf,
            year_range=year_range,
        )

        bm_change_brackets = process_data.dectect_changes(dataset=timeseries_container, variable="basal_melt_flux")

        if len(bm_change_brackets) == 1:
            main = html.P(
                "There have been no significant changes in basal melt flux detected in the selcted period.",
                className="insights_calving_summary_text",
            )
        else:
            last_change = timeseries_container.timeseries["basal_melt_flux"].measurements.index[bm_change_brackets[-2]]
            main = html.P(
                f"There have been {num2words(len(bm_change_brackets) - 1)} periods where the basal melt flux has changed in the selected time span over {ice_shelf} Ice Shelf. The most recent was in {last_change}.",
                className="insights_calving_summary_text",
            )

        text = [
            main,
        ]
        return html.Div(text)
    return html.Div()


@callback(
    Output("basal_graph", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def plot_basal_melt_flux(ice_shelf, year_range):
    if ice_shelf is None:
        ice_shelf = "thwaites"

    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    basal_events = process_data.detect_outlier_events(
        timeseries_container=timeseries_container, variable="basal_melt_flux", stationary_check=False
    )

    bm_change_brackets = process_data.dectect_changes(dataset=timeseries_container, variable="basal_melt_flux")

    fig = plotting.plot_summary_timeseries(
        timeseries_container.timeseries["basal_melt_flux"],
        outliers=(basal_events[4], basal_events[2]),
        brackets=bm_change_brackets,
        margin={"t": 0, "l": 150, "b": 25, "r": 50},
        y_label="Basal Melt Flux (Gt/year)",
    )
    return fig


@callback(
    Output("insights_basal_detail_div", "style"),
    Output("basal_graph_div", "style"),
    Input("insights_btn_show_basal", "n_clicks"),
)
def button_detailed_basal_melt_flux(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}


@callback(
    Output("insights_headline_title_cov_analysis_div", "children"),
    Output("insights_headline_text_cov_analysis_div", "children"),
    Output("insights_btn_show_cov_analysis_div", "style"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_cov_analysis_headline(ice_shelf, year_range):
    if ice_shelf == "thwaites":
        strong_correlation, medium_correlation, weak_correlation = process_data.example_Covariate_analysis_thwaites(
            ice_shelf_polygons_epsg4326, ocean_polygons_epsg4326, config_dict, year_range
        )

        total_correlations = len(strong_correlation) + len(medium_correlation) + len(weak_correlation)

        if total_correlations == 0:
            return html.Div(), html.Div(), {"display": "none"}
        return (
            html.Div(html.P(html.Strong("Covariate Analysis:"), className="insights_headline_text")),
            html.Div(
                html.P(
                    f"There have been {total_correlations} correlated variables detected.",
                    className="insights_headline_text",
                )
            ),
            {
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
            },
        )
    return html.Div(), html.Div(), {"display": "none"}


@callback(
    Output("insights_cov_analysis_detail_div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def insight_update_cov_analysis_detail(ice_shelf, year_range):
    if ice_shelf == "thwaites":
        strong_correlation, medium_correlation, weak_correlation = process_data.example_Covariate_analysis_thwaites(
            ice_shelf_polygons_epsg4326, ocean_polygons_epsg4326, config_dict, year_range
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
                        f"{' '.join(corr_vars[0].lower().capitalize().split('_')[:-1])} and {' '.join(corr_vars[1].lower().split('_')[:-1])} (correlation coefficient = {strong_correlation[corr_vars][0]: .2f}, p-value = {strong_correlation[corr_vars][1]: .3f})",
                        className="insights_calving_summary_text_no_just_indent",
                    )
                )
        else:
            strong_corr_text = html.P("", className="insights_calving_summary_text_no_just_indent")

        if len(medium_correlation) > 0:
            medium_corr_text = html.P(
                " The following medium correlations have been observed: ",
                className="insights_calving_summary_text_no_just",
            )
            for corr_vars in medium_correlation:
                medium_corr_text_ul.append(
                    html.P(
                        f"{' '.join(corr_vars[0].lower().capitalize().split('_')[:-1])} and {' '.join(corr_vars[1].lower().split('_')[:-1])} (correlation coefficient = {medium_correlation[corr_vars][0]: .2f}, p-value = {medium_correlation[corr_vars][1]: .3f})",
                        className="insights_calving_summary_text_no_just_indent",
                    )
                )
        else:
            medium_corr_text = html.P("", className="insights_calving_summary_text_no_just_indent")

        # medium_corr_text = html.P(medium_corr_text, className="insights_calving_summary_text_no_just")

        if len(weak_correlation) > 0:
            weak_corr_text = html.P(
                " The following weak correlations have been observed: ",
                className="insights_calving_summary_text_no_just",
            )
            for corr_vars in weak_correlation:
                weak_corr_text_ul.append(
                    html.P(
                        f"{' '.join(corr_vars[0].lower().capitalize().split('_')[:-1])} and {' '.join(corr_vars[1].lower().split('_')[:-1])} (correlation coefficient = {weak_correlation[corr_vars][0]: .2f}, p-value = {weak_correlation[corr_vars][1]: .3f})",
                        className="insights_calving_summary_text_no_just_indent",
                    )
                )
        else:
            weak_corr_text = html.P("", className="insights_calving_summary_text_no_just_indent")

        text = (
            [strong_corr_text]
            + strong_corr_text_ul
            + [medium_corr_text]
            + medium_corr_text_ul
            + [weak_corr_text]
            + weak_corr_text_ul
        )
        return html.Div(text)
    return html.Div()


@callback(
    Output("cov_analysis_graph_div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def plot_cov_analysis(ice_shelf, year_range):
    ice_shelf = "thwaites"
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

    variables = [
        "moorings_1d-depth_PIG_N_and_S_combined",
        "basal_melt_anomalies_pine_island_1d-basal_melt_anomalies",
        "subglacial_lake_discharge_1d-lake_discharge",
        # "subglacial_lakes_volume_change_1d-volume_change",
        "basal_melt_anomalies_thwaites_1d-basal_melt_anomalies",
    ]

    axis_labels = []
    for variable in variables:
        # print(variable)
        axis_labels.append(
            config_dict[variable.split("-")[0].lower()]["variables"][variable.split("-")[-1]]["y_axis_label"]
        )

    timeseries = query_data.process_all_data(variables, polygons, year_range)

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=timeseries,
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
    )

    analysis_type_enum = JointPlotAnalysisType.CORR
    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
    fig, _ = covariate_analyser.covariate_analysis(max_lag=3, axis_labels=axis_labels, analysis_type=analysis_type_enum)
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image_string = base64.b64encode(buffer.getvalue()).decode()

    results_plot = "data:image/png;base64," + image_string
    results_plot = html.Img(src=results_plot, style={"max-width": "100%", "max-height": "100%"})

    return results_plot


@callback(
    Output("insights_cov_analysis_detail_div", "style"),
    Output("cov_analysis_graph_div", "style"),
    Input("insights_btn_show_cov_analysis", "n_clicks"),
)
def button_detailed_cov_analysis(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"text-align": "center", "display": "block"}
    return {"display": "none"}, {"text-align": "center", "display": "none"}


# These need to be refactored to be a pattern matching callback so there is only one for all of the switches.
@callback(
    Output(f"switch-group-vars-{'polarice_elevation_change'.lower()}", "style"),
    Input(f"ds_{'polarice_elevation_change'.lower()}", "value"),
)
def elev_polarice_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'elevation_change_eolis'.lower()}", "style"),
    Input(f"ds_{'elevation_change_eolis'.lower()}", "value"),
)
def elev_eolis_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


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
    Output(f"switch-group-vars-{'its_live_ice_velocity'.lower()}", "style"),
    Input(f"ds_{'its_live_ice_velocity'.lower()}", "value"),
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
    Output(f"switch-group-vars-{'GLORYS12V1'}", "style"),
    Input(f"ds_{'GLORYS12V1'}", "value"),
)
def glorys12v1_visable(visability):
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
    Output(f"switch-group-vars-{'moorings_1d'.lower()}", "style"),
    Input(f"ds_{'moorings_1d'.lower()}", "value"),
)
def thwaites_moorings_1d_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'subglacial_lake_discharge_1d'.lower()}", "style"),
    Input(f"ds_{'subglacial_lake_discharge_1d'.lower()}", "value"),
)
def thwaites_subglacial_lake_discharge_1d_visable(visability):
    if visability:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output(f"switch-group-vars-{'subglacial_lakes_volume_change_1d'.lower()}", "style"),
    Input(f"ds_{'subglacial_lakes_volume_change_1d'.lower()}", "value"),
)
def thwaites_subglacial_lakes_volume_change_1d_visable(visability):
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
    summary_data, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    total_mass_loss, total_balanace_unc = process_data.generate_statistic(
        timeseries_container=timeseries_container, variable="mass_balance"
    )

    cummulative_retreat, cummulative_retreat_unc = process_data.generate_statistic_cummulative_difference(
        timeseries_container=timeseries_container, variable="cumulative_retreat", year_range=year_range
    )

    total_basal_melt_flux, total_basal_melt_flux_unc = process_data.generate_statistic(
        timeseries_container=timeseries_container, variable="basal_melt_flux"
    )
    total_calving, total_calving_unc = process_data.generate_statistic(
        timeseries_container=timeseries_container, variable="calving"
    )

    total_basal_melt_contribution, total_basal_melt_contribution_unc = process_data.generate_statistic(
        timeseries_container=timeseries_container,
        variable="basal_melt_flux",
        steady_state_val=summary_data["steady_state_basal_melt"].values[0][0],
    )

    lost_gained = "lost" if total_mass_loss < 0 else "gained"
    loss_gain = "loss" if total_mass_loss < 0 else "gain"

    return html.Div(
        [
            html.P(" "),
            html.H5(f"{' '.join(ice_shelf.split('_')).title()} Ice Shelf Summary:"),
            html.P(
                f"Overall the {' '.join(ice_shelf.split('_')).title()} Ice Shelf has {lost_gained} {int(np.round(np.abs(total_mass_loss))): .0f} Gt of ice between {year_range[0]} and {year_range[1]}.",
                style={"padding-right": "100px", "padding-top": "10px"},
            ),
            html.Ul(
                [
                    html.Li(
                        f"Basal melting has contributed approximately {np.abs(total_basal_melt_contribution / (year_range[1] - year_range[0])): .0f} Gt/yr to this total {loss_gain} over this period."
                    ),
                    html.Li(
                        f"Calving front retreat has contributed approximately {np.abs(cummulative_retreat / (year_range[1] - year_range[0])):.0f} Gt/yr to this total {loss_gain} over this period."
                    ),
                ]
            ),
            html.P(
                "Fresh water flux to the ocean is partitioned as:",
                style={"padding-right": "100px", "padding-top": "0px"},
            ),
            html.Ul(
                [
                    html.Li(f"{np.abs(total_calving):.0f} Gt from iceberg calving."),
                    html.Li(f"{np.abs(total_basal_melt_flux):.0f} Gt from basal melting."),
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
    Input("summary_year_selector", "value"),
    State("gauge_1_store", "data"),
)
def plot_gauge_indicator1(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    fig = plotting.plot_gauge_indicator_timeseries(timeseries_container, variable)
    return fig


@callback(
    Output("total_mass_change_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator1_text(year_range):
    return html.Div(
        html.P(
            [
                "Total mass change between",
                html.Br(),
                f"{year_range[0]} and {year_range[1]} (Gt)",
            ]
        ),
    )


@callback(
    Output("gauge_2_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("gauge_2_store", "data"),
)
def plot_gauge_indicator2(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    fig = plotting.plot_gauge_indicator_timeseries(timeseries_container, variable)
    return fig


@callback(
    Output("total_basal_melt_flux_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator2_text(year_range):
    return html.Div(
        html.P(
            [
                "Total basal melting between",
                html.Br(),
                f"{year_range[0]} and {year_range[1]} (Gt)",
            ]
        ),
    )


@callback(
    Output("gauge_3_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("gauge_3_store", "data"),
)
def plot_gauge_indicator3(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    fig = plotting.plot_gauge_indicator_timeseries(timeseries_container, variable)
    return fig


@callback(
    Output("total_calving_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator3_text(year_range):
    return html.Div(
        html.P(
            [
                "Total calving between",
                html.Br(),
                f"{year_range[0]} and {year_range[1]} (Gt)",
            ]
        ),
    )


@callback(
    Output("gauge_4_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("gauge_4_store", "data"),
)
def plot_gauge_indicator4(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    fig = plotting.plot_gauge_indicator_timeseries(timeseries_container, variable)
    return fig


@callback(
    Output("total_discharge_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator4_text(year_range):
    return html.Div(
        html.P(
            [
                "Total discharge between",
                html.Br(),
                f"{year_range[0]} and {year_range[1]} (Gt)",
            ]
        ),
    )


@callback(
    Output("gauge_5_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("gauge_5_store", "data"),
)
def plot_gauge_indicator5(ice_shelf, year_range, variable):
    _, timeseries_container, _, _ = query_data.query_summary_data(
        ice_shelf_polygons=ice_shelf_polygons_proj,
        ice_shelf=ice_shelf,
        year_range=year_range,
    )

    fig = plotting.plot_gauge_indicator_timeseries(timeseries_container, variable)
    return fig


@callback(
    Output("total_surface_mass_balance_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator5_text(year_range):
    return html.Div(
        html.P(
            [
                "Total surface mass balance between",
                html.Br(),
                f"{year_range[0]} and {year_range[1]} (Gt)",
            ]
        ),
    )


# @callback(
#     Output("results-plot", "children"),
#     Input("run_analysis-button", "n_clicks"),
#     State("ice_shelf_selection", "value"),
#     State("summary_year_selector", "value"),
#     State("select-analysis-type", "value"),
#     State({"type": "data_selector", "index": ALL}, "id"),
#     State({"type": "data_selector", "index": ALL}, "value"),
# )
# def conduct_covariate_analysis(
#     n,
#     ice_shelf,
#     year_range,
#     analysis_type,
#     dataset_ids,
#     datasets,
# ):
#     ice_shelf_extent = ice_shelf_polygons_epsg4326[ice_shelf_polygons_epsg4326["Name"] == ice_shelf]["geometry"].values[
#         0
#     ]
#     ocean_extent = ocean_polygons_epsg4326[ocean_polygons_epsg4326["Name"] == ice_shelf]["geometry"].values[0]
#     climate_extent = None

#     # polygons = {"ice_shelf": ice_shelf_extent, "ocean": ocean_extent, "climate": climate_extent}

#     polygons = gpd.GeoDataFrame(
#         {"Name": ["ice_shelf", "ocean", "climate"], "geometry": [ice_shelf_extent, ocean_extent, climate_extent]}
#     )
#     polygons = polygons.set_crs("epsg: 4326").set_index("Name")

#     variables = [dataset_ids[i]["index"] for i, dataset in enumerate(datasets) if dataset]

#     axis_labels = []
#     for variable in variables:
#         # print(variable)
#         axis_labels.append(
#             config_dict[variable.split("-")[0].lower()]["variables"][variable.split("-")[-1]]["y_axis_label"]
#         )

#     timeseries = query_data.process_all_data(variables, polygons, year_range)

#     if len(timeseries) > 0:
#         timeseries_container = TimeSeriesPreprocessor(
#             timeseries=timeseries,
#             time_period=AggregationPeriod.MONTH,
#             aggregation=AggregationStrategy.MEAN,
#         )

#         if analysis_type == "1":
#             analysis_type_enum = JointPlotAnalysisType.CORR
#         elif analysis_type == "2":
#             analysis_type_enum = JointPlotAnalysisType.CROSS_CORR
#         elif analysis_type == "3":
#             analysis_type_enum = JointPlotAnalysisType.CAUSAL
#         covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
#         fig, axes = covariate_analyser.covariate_analysis(
#             max_lag=6, axis_labels=axis_labels, analysis_type=analysis_type_enum
#         )
#         buffer = BytesIO()
#         fig.savefig(buffer, format="png")
#         buffer.seek(0)
#         image_string = base64.b64encode(buffer.getvalue()).decode()

#         results_plot = "data:image/png;base64," + image_string
#         results_plot = html.Img(src=results_plot, style={"max-width": "100%", "max-height": "100%"})

#         return results_plot
#     return None


# @callback(
#     Output("analysis_results_div", "children"),
#     Input("run_analysis-button", "n_clicks"),
#     State("ice_shelf_selection", "value"),
#     State("summary_year_selector", "value"),
#     State("select-analysis-type", "value"),
#     State({"type": "data_selector", "index": ALL}, "id"),
#     State({"type": "data_selector", "index": ALL}, "value"),
# )
# def conduct_covariate_analysis_text(
#     n,
#     ice_shelf,
#     year_range,
#     analysis_type,
#     dataset_ids,
#     datasets,
# ):
#     ice_shelf_extent = ice_shelf_polygons_epsg4326[ice_shelf_polygons_epsg4326["Name"] == ice_shelf]["geometry"].values[
#         0
#     ]
#     ocean_extent = ocean_polygons_epsg4326[ocean_polygons_epsg4326["Name"] == ice_shelf]["geometry"].values[0]
#     climate_extent = None

#     # polygons = {"ice_shelf": ice_shelf_extent, "ocean": ocean_extent, "climate": climate_extent}

#     polygons = gpd.GeoDataFrame(
#         {"Name": ["ice_shelf", "ocean", "climate"], "geometry": [ice_shelf_extent, ocean_extent, climate_extent]}
#     )
#     polygons = polygons.set_crs("epsg: 4326").set_index("Name")

#     variables = [dataset_ids[i]["index"] for i, dataset in enumerate(datasets) if dataset]
#     # variab = [dataset_ids[i]["index"] for i, dataset in enumerate(datasets) if dataset]
#     # variables = [
#     #     "moorings_1d-depth_PIG_N_and_S_combined",
#     #     "basal_melt_anomalies_pine_island_1d-basal_melt_anomalies",
#     #     "subglacial_lakes_volume_change_1d-volume_change",
#     #     "basal_melt_anomalies_thwaites_1d-basal_melt_anomalies",
#     # ]

#     axis_labels = []
#     for variable in variables:
#         # print(variable)
#         axis_labels.append(
#             config_dict[variable.split("-")[0].lower()]["variables"][variable.split("-")[-1]]["y_axis_label"]
#         )

#     timeseries = query_data.process_all_data(variables, polygons, year_range)

#     if len(timeseries) > 0:
#         timeseries_container = TimeSeriesPreprocessor(
#             timeseries=timeseries,
#             time_period=AggregationPeriod.MONTH,
#             aggregation=AggregationStrategy.MEAN,
#         )

#         covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

#         if analysis_type == "1":
#             strong_correlation = covariate_analyser.find_correlated_vars(corr_thresh=0.8)
#             medium_correlation = covariate_analyser.find_correlated_vars(corr_thresh=0.5)
#             weak_correlation = covariate_analyser.find_correlated_vars(corr_thresh=0.25)
#         elif analysis_type == "2":
#             strong_correlation = covariate_analyser.find_crosscorrelated_vars(corr_thresh=0.8, max_lag=6)
#             medium_correlation = covariate_analyser.find_crosscorrelated_vars(corr_thresh=0.5, max_lag=6)
#             weak_correlation = covariate_analyser.find_crosscorrelated_vars(corr_thresh=0.25, max_lag=6)
#         elif analysis_type == "3":
#             strong_correlation = covariate_analyser.find_causal_vars()
#             medium_correlation = covariate_analyser.find_causal_vars()
#             weak_correlation = covariate_analyser.find_causal_vars()

#         for key in strong_correlation:
#             del medium_correlation[key]
#             del weak_correlation[key]

#         for key in medium_correlation:
#             del weak_correlation[key]

#         strong_corr_text_ul = []
#         medium_corr_text_ul = []
#         weak_corr_text_ul = []

#         if len(strong_correlation) > 0:
#             strong_corr_text = html.P(
#                 " The following strong correlations have been observed: ",
#                 className="analysis_results",
#             )
#             for corr_vars in strong_correlation:
#                 strong_corr_text_ul.append(
#                     html.P(
#                         f"{
#                             config_dict[corr_vars[0].split('-')[0].lower()]['variables'][corr_vars[0].split('-')[1]][
#                                 'label'
#                             ]
#                         } and {
#                             config_dict[corr_vars[1].split('-')[0].lower()]['variables'][corr_vars[1].split('-')[1]][
#                                 'label'
#                             ]
#                         } (correlation coefficient = {strong_correlation[corr_vars][0]: .2f}, p-value = {
#                             strong_correlation[corr_vars][1]: .3f})",
#                         # f"{' '.join(corr_vars[0].lower().capitalize().split('_'))} and {' '.join(corr_vars[1].lower().split('_'))} (correlation coefficient = {strong_correlation[corr_vars][0]: .2f}, p-value = {strong_correlation[corr_vars][1]: .3f})",
#                         className="analysis_results_indent",
#                     )
#                 )
#         else:
#             strong_corr_text = html.P("", className="insights_calving_summary_text_no_just_indent")

#         if len(medium_correlation) > 0:
#             medium_corr_text = html.P(
#                 " The following medium correlations have been observed: ",
#                 className="analysis_results",
#             )
#             for corr_vars in medium_correlation:
#                 medium_corr_text_ul.append(
#                     html.P(
#                         f"{
#                             config_dict[corr_vars[0].split('-')[0].lower()]['variables'][corr_vars[0].split('-')[1]][
#                                 'label'
#                             ]
#                         } and {
#                             config_dict[corr_vars[1].split('-')[0].lower()]['variables'][corr_vars[1].split('-')[1]][
#                                 'label'
#                             ]
#                         } (correlation coefficient = {medium_correlation[corr_vars][0]: .2f}, p-value = {
#                             medium_correlation[corr_vars][1]: .3f})",
#                         className="analysis_results_indent",
#                     )
#                 )
#         else:
#             medium_corr_text = html.P("", className="analysis_results_indent")

#         if len(weak_correlation) > 0:
#             weak_corr_text = html.P(
#                 " The following weak correlations have been observed: ",
#                 className="analysis_results",
#             )
#             for corr_vars in weak_correlation:
#                 weak_corr_text_ul.append(
#                     html.P(
#                         f"{
#                             config_dict[corr_vars[0].split('-')[0].lower()]['variables'][corr_vars[0].split('-')[1]][
#                                 'label'
#                             ]
#                         } and {
#                             config_dict[corr_vars[1].split('-')[0].lower()]['variables'][corr_vars[1].split('-')[1]][
#                                 'label'
#                             ]
#                         } (correlation coefficient = {weak_correlation[corr_vars][0]: .2f}, p-value = {
#                             weak_correlation[corr_vars][1]: .3f})",
#                         className="analysis_results_indent",
#                     )
#                 )
#         else:
#             weak_corr_text = html.P("", className="analysis_results_indent")

#         text = (
#             [html.H5("Analysis Results:", className="analysis_heading5")]
#             + [strong_corr_text]
#             + strong_corr_text_ul
#             + [medium_corr_text]
#             + medium_corr_text_ul
#             + [weak_corr_text]
#             + weak_corr_text_ul
#         )
#         return html.Div(text)
#     return html.Div()

# Output("results-plot", "children"),
# Input("run_analysis-button", "n_clicks"),
# State("ice_shelf_selection", "value"),
# State("summary_year_selector", "value"),
# State("select-analysis-type", "value"),
# State({"type": "data_selector", "index": ALL}, "id"),
# State({"type": "data_selector", "index": ALL}, "value"),


@callback(
    Output("results-plot", "src"),
    Output("error-message", "is_open", allow_duplicate=True),
    Output("error-message", "children", allow_duplicate=True),
    Input("run_analysis-button", "n_clicks"),
    State("ice_shelf_selection", "value"),
    State("summary_year_selector", "value"),
    State("select-analysis-type", "value"),
    State({"type": "data_selector", "index": ALL}, "id"),
    State({"type": "data_selector", "index": ALL}, "value"),
    running=[
        (Output("run_analysis-button", "disabled"), True, False),
        (Output("status", "children"), "Setting up...", "Done!"),
        (Output("progress", "value"), 0, 100),
        (Output("job-status", "is_open"), True, False),
    ],
    progress=[
        Output("status", "children"),
        Output("progress", "value"),
    ],
    manager=background_callback_manager,
    background=True,
    prevent_initial_call=True,
)
# n,
#     ice_shelf,
#     year_range,
#     analysis_type,
#     dataset_ids,
#     datasets,
def conduct_covariate_analysis_integrated(
    set_progress, n_clicks, ice_shelf, year_range, analysis_type, dataset_ids, datasets
):
    # logger.info(start_time)
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

    variables = [
        {
            "name": f"{dataset_ids[i]['index'].split('-')[0]}: {dataset_ids[i]['index'].split('-')[1]}",
            "dataset": dataset_ids[i]["index"].split("-")[0],
            "variable": dataset_ids[i]["index"].split("-")[1],
            "extent": config_dict[dataset_ids[i]["index"].split("-")[0]]["variables"][
                dataset_ids[i]["index"].split("-")[1]
            ]["region"],
        }
        for i, dataset in enumerate(datasets)
        if dataset
    ]

    print(variables)
    # variables = [dataset_ids[i]["index"] for i, dataset in enumerate(datasets) if dataset]

    try:
        res = requests.post(
            f"{url}/ice-shelves/{ice_shelf}/covariate-analysis",
            json={
                "start_time": pd.Timestamp(year=year_range[0], month=1, day=1).isoformat(),
                "end_time": pd.Timestamp(year=year_range[1], month=1, day=1).isoformat(),
                "variables": variables,
            },
        )
        res.raise_for_status()
        job_id = res.json()["job_id"]
        set_props("job-id", {"data": job_id})

        while True:
            sleep(1)
            res = requests.get(f"{url}/jobs/{job_id}")
            res.raise_for_status()
            data = res.json()

            if data["status"] == "Failed":
                return "", True, "Job failed or was cancelled."
            if data["status"] == "Error":
                return "", True, "An error occurred while running the job."

            pdone = data["progress_done"]
            ptotal = data["progress_total"]

            percent = 5 if ptotal == 0 else max(5, min(95, int(pdone / ptotal * 100)))

            set_progress(
                (
                    data["status"] + "...",
                    percent,
                )
            )
            if ptotal > 0 and pdone >= ptotal:
                break

        set_progress(("Downloading result...", 95))

        output_url = data["outputs"]["use-case-3-covariate-analyser"]["output_plot"]
        bucket, key = output_url.lower().removeprefix("s3://").split("/", 1)
        b = BytesIO()
        s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=b)
        b.seek(0)
        src = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception as e:
        logger.exception("Error occurred while generating plot")
        if "res" in locals():
            logger.error(res.text)
        return "", True, str(e)
    return src, False, ""


@callback(
    Output("cancel-button", "disabled"),
    Output("error-message", "is_open", allow_duplicate=True),
    Output("error-message", "children", allow_duplicate=True),
    Input("cancel-button", "n_clicks"),
    State("job-id", "data"),
    prevent_initial_call=True,
)
def cancel_job(n_clicks, job_id):
    if not job_id:
        return True, False, ""
    if not n_clicks:
        return False, False, ""
    try:
        res = requests.delete(f"{url}/jobs/{job_id}")
        res.raise_for_status()
    except Exception as e:
        logger.error(e)
        return False, True, str(e)
    return True, False, ""


if __name__ == "__main__":
    app.run(debug=True)
