"""Provides report style data insights for an ice shelf."""

from datetime import datetime

import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html, register_page
from num2words import num2words

from use_case_3.dashboard.components.ice_shelf_selection import create_ice_shelf_map, create_ice_shelf_selection
from use_case_3.dashboard.dashboard_layout import layout_navbar
from use_case_3.dashboard.plotting import plot_summary_timeseries, plot_summary_timeseries_multiple
from use_case_3.dashboard.process_data import (
    calculate_mass_balance,
    calculate_median_change_brackets,
    dectect_changes,
    detect_outlier_events,
    example_covariate_analysis_thwaites,
    example_subglacial_lake_analysis,
)
from use_case_3.dashboard.query_api import query_summary_data

register_page(__name__)

ice_shelf_inital = None

navbar = layout_navbar("/insights")

layout = html.Div(
    [
        navbar,
        dbc.Row(
            [
                dbc.Col(
                    [
                        # html.Div(
                        #     html.Img(src="assets/DigitalTwinLogoBlueFINAL13052025.png", className="insights_picture"),
                        # ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.P(
                            (
                                "Welcome to Digital Twin Ice Sheets. Use the menu below to select an ice shelf for "
                                "diagnostics of its present state and observed changes over time, derived from "
                                "satellite observations, in-situ measurements, models, and reanalysis data."
                            ),
                            className="insights_intro_text",
                        ),
                        *create_ice_shelf_selection(show_map=False),
                        dbc.Row(
                            html.Div(html.H1("Individual Variables: ")),
                            style={"padding-top": "20px"},
                        ),
                        # Insights mass balance
                        dbc.Row(
                            [
                                dbc.Col([html.Div(id="insights_headline_title_mass_balance_div")], width=3),
                                dbc.Col(
                                    [
                                        html.Div(
                                            id="insights_headline_text_mass_balance_div",
                                        ),
                                    ],
                                    width=9,
                                ),
                            ]
                        ),
                        dbc.Row(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Show / Hide Detailed Mass Balance Analysis",
                                        color="primary",
                                        id="insights_btn_show_mass_balance",
                                    ),
                                ],
                                style={"display": "none"},
                                id="insights_btn_show_mass_balance_div",
                            ),
                        ),
                        html.Div(id="insights_mass_balance_div", style={"display": "none"}),
                        html.Div(
                            dcc.Graph(id="mass_balance_graph"), id="mass_balance_graph_div", style={"display": "none"}
                        ),
                        # Insights calving
                        dbc.Row(
                            [
                                dbc.Col([html.Div(id="insights_headline_title_calving_div")], width=3),
                                dbc.Col(
                                    [
                                        html.Div(
                                            id="insights_headline_text_calving_div",
                                        ),
                                    ],
                                    width=9,
                                ),
                            ]
                        ),
                        dbc.Row(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Show / Hide Detailed Calving Analysis",
                                        color="primary",
                                        id="insights_btn_show_calving",
                                    ),
                                ],
                                style={"display": "none"},
                                id="insights_btn_show_calving_div",
                            ),
                        ),
                        html.Div(
                            id="insights_calving_detail_div",
                        ),
                        html.Div(dcc.Graph(id="calving_graph"), style={"display": "none"}, id="calving_graph_div"),
                        # Insights basal melting
                        dbc.Row(
                            [
                                dbc.Col([html.Div(id="insights_headline_title_basal_div")], width=3),
                                dbc.Col(
                                    [
                                        html.Div(
                                            id="insights_headline_text_basal_div",
                                        ),
                                    ],
                                    width=9,
                                ),
                            ]
                        ),
                        dbc.Row(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Show / Hide Detailed Basal Melt Flux Analysis",
                                        color="primary",
                                        id="insights_btn_show_basal",
                                    ),
                                ],
                                style={"display": "none"},
                                id="insights_btn_show_basal_div",
                            ),
                        ),
                        html.Div(
                            id="insights_basal_detail_div",
                        ),
                        html.Div(dcc.Graph(id="basal_graph"), style={"display": "none"}, id="basal_graph_div"),
                        # Insights subglacial lakes
                        dbc.Row(
                            [
                                dbc.Col([html.Div(id="insights_headline_title_sg_lakes_div")], width=3),
                                dbc.Col(
                                    [
                                        html.Div(
                                            id="insights_headline_text_sg_lakes_div",
                                        ),
                                    ],
                                    width=9,
                                ),
                            ]
                        ),
                        dbc.Row(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Show / Hide Detailed Subglacial Lake Analysis",
                                        color="primary",
                                        id="insights_btn_show_sg_lakes",
                                    ),
                                ],
                                style={"display": "none"},
                                id="insights_btn_show_sg_lakes_div",
                            ),
                        ),
                        html.Div(
                            id="insights_sg_lakes_detail_div",
                        ),
                        html.Div(
                            dcc.Graph(id="sg_lakes_graph_vol_change"),
                            style={"display": "none"},
                            id="sg_lakes_graph_vol_change_div",
                        ),
                        html.Div(
                            dcc.Graph(id="sg_lakes_graph_rate_vol_change"),
                            style={"display": "none"},
                            id="sg_lakes_graph_rate_vol_change_div",
                        ),
                        dbc.Row(
                            html.Div(html.H1("Correlated Variables: "), id="insights_correlated_variables_title_div"),
                            style={"padding-top": "20px"},
                        ),
                        # Insights Covariate Analysis
                        dbc.Row(
                            [
                                dbc.Col([html.Div(id="insights_headline_title_cov_analysis_div")], width=3),
                                dbc.Col(
                                    [
                                        html.Div(
                                            id="insights_headline_text_cov_analysis_div",
                                        ),
                                    ],
                                    width=9,
                                ),
                            ]
                        ),
                        dbc.Row(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Show / Hide Detailed Analysis",
                                        color="primary",
                                        id="insights_btn_show_cov_analysis",
                                    ),
                                ],
                                style={"display": "none"},
                                id="insights_btn_show_cov_analysis_div",
                            ),
                        ),
                        html.Div(
                            id="insights_cov_analysis_detail_div",
                        ),
                        html.Div(
                            id="cov_analysis_graph_div",
                            style={"text-align": "center", "display": "none"},
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [create_ice_shelf_map()],
                    width=4,
                ),
            ],
            style={"margin": "0"},
        ),
    ]
)


@callback(
    Output("insights_correlated_variables_title_div", "style"),
    Input("ice-shelf-selection", "value"),
)
def hide_insights_correlated_vars_title(ice_shelf_selection):  # noqa: ANN001, ANN201, D103
    if ice_shelf_selection == "thwaites":
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("insights_headline_title_mass_balance_div", "children"),
    Output("insights_headline_text_mass_balance_div", "children"),
    Output("insights_btn_show_mass_balance_div", "style"),
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_massbalance_headline(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is not None:
        _, timeseries_container = await query_summary_data(
            ice_shelf=ice_shelf,
            start_year=year_range[0],
            end_year=year_range[1],
        )

        mass_balance_events, _, _ = calculate_mass_balance(timeseries_container=timeseries_container)

        gained_lost = "lost" if mass_balance_events[0] < 0 else "gained"
        return (
            html.Div(html.P(html.Strong("MASS BALANCE:"), className="insights_headline_text")),
            html.Div(
                html.P(
                    (
                        f"In this {year_range[1] - year_range[0]} year period, {ice_shelf.replace('_', ' ').title()} "
                        f"ice shelf has {gained_lost} {np.abs(mass_balance_events[0]): .0f} Gt of ice."
                    ),
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
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_massbalance_detail(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is not None:
        _, timeseries_container = await query_summary_data(
            ice_shelf=ice_shelf,
            start_year=year_range[0],
            end_year=year_range[1],
        )
        mass_balance_events, mass_balance_rate, calving_matched_years = calculate_mass_balance(
            timeseries_container=timeseries_container
        )

        gained_lost = "gained" if mass_balance_events[0] > 0 else "lost"
        increasing_decreasing = "increasing" if mass_balance_rate < 0 else "decreasing"

        if len(calving_matched_years) > 0:
            calving_matched_text = (
                f" It is likely that {num2words(len(calving_matched_years))} of these events are due to calving."
            )
        else:
            calving_matched_text = " Currently there are no obvious causes of these events"
        # title = html.H6("Mass Balance:", className="insights_h6")
        # intro = html.P(
        intro = dcc.Markdown(
            (
                f"In the past {year_range[1] - year_range[0]} years, {ice_shelf.replace('_', ' ').title()} "
                f"ice shelf has {gained_lost} {np.abs(mass_balance_events[0]): .0f}+/-{mass_balance_events[1]: .0f}Gt "
                f"of ice. This mass change is currently {increasing_decreasing} at a rate of "
                f"{np.abs(mass_balance_rate): .1f} Gt/yr <sup>2</sup>."
            ),
            dangerously_allow_html=True,
            className="insights_calving_summary_text",
        )
        main = html.P()
        if len(mass_balance_events[2]) > 0:
            if len(mass_balance_events[2]) == 1:
                main = html.P(
                    (
                        f"There was one significant event on {ice_shelf.replace('_', ' ').title()} "
                        f"Ice shelf between {year_range[0]} and {year_range[1]}. "
                        f"This event occured in {mass_balance_events[4][-1]}, and resulted an overall ice loss of "
                        f"{mass_balance_events[2][-1]: .0f}+/-{mass_balance_events[3][-1]: .0f}Gt."
                    ),
                    className="insights_calving_summary_text",
                )
            else:
                main = html.P(
                    (
                        f"There were {num2words(len(mass_balance_events[2]))} significant events "
                        f"impacting {ice_shelf.replace('_', ' ').title()} ice shelf mass balance "
                        f"between {year_range[0]} and {year_range[1]}." + calving_matched_text,
                    ),
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
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def plot_mass_balance(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is None:
        ice_shelf = "thwaites"
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    mass_balance_events, _, _ = calculate_mass_balance(timeseries_container=timeseries_container)

    return plot_summary_timeseries(
        timeseries_container.timeseries["mass_balance"],
        outliers=(mass_balance_events[4], mass_balance_events[2]),
        margin={"t": 0, "l": 150, "b": 25, "r": 50},
        y_label="Mass Balance (Gt/year)",
    )


@callback(
    Output("insights_mass_balance_div", "style"),
    Output("mass_balance_graph_div", "style"),
    Input("insights_btn_show_mass_balance", "n_clicks"),
)
def button_detailed_mass_balance(n_clicks):  # noqa: ANN001, ANN201, D103
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}


@callback(
    Output("insights_headline_title_calving_div", "children"),
    Output("insights_headline_text_calving_div", "children"),
    Output("insights_btn_show_calving_div", "style"),
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_calving_headline(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is not None:
        _, timeseries_container = await query_summary_data(
            ice_shelf=ice_shelf,
            start_year=year_range[0],
            end_year=year_range[1],
        )

        _, _, calving_events, _, _ = detect_outlier_events(timeseries_container=timeseries_container)
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
                        (
                            f"There {has_have} been {num2words(len(calving_events))} {year_s} "
                            f"with large calving {event_s} in the {year_range[1] - year_range[0]} year period chosen."
                        ),
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
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_calving_detail(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is not None:
        _, timeseries_container = await query_summary_data(
            ice_shelf=ice_shelf,
            start_year=year_range[0],
            end_year=year_range[1],
        )

        total, total_unc, calving_events, calving_unc_events, time_events = detect_outlier_events(
            timeseries_container=timeseries_container
        )
        if len(calving_events) == 0:
            return html.Div()
        if len(calving_events) == 1:
            main = html.P(
                (
                    f"There was one significant calving event on {ice_shelf.replace('_', ' ').title()} "
                    f"Ice shelf between {year_range[0]} and {year_range[1]}. "
                    f"This calving event occured in {time_events[0]}, and resulted an overall ice loss "
                    f"of {calving_events[0]: .0f}+/-{calving_unc_events[0]: .0f}Gt."
                ),
                className="insights_calving_summary_text",
            )
        else:
            main = html.P(
                (
                    f"There were {num2words(len(calving_events))} significant calving events on "
                    f"{ice_shelf.replace('_', ' ').title()} Ice shelf between {year_range[0]} and {year_range[1]}. "
                    f"These calving events have resulted in an overal ice loss of approximately "
                    f"{total: .0f}+/-{total_unc: .0f}Gt. The largest calving event was "
                    f"{calving_events[-1]: .0f}+/-{calving_unc_events[-1]: .0f}Gt and occured in {time_events[-1]}."
                ),
                className="insights_calving_summary_text",
            )

        text = [
            main,
        ]
        return html.Div(text)
    return html.Div()


@callback(
    Output("calving_graph", "figure"),
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def plot_calving(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is None:
        ice_shelf = "thwaites"

    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    calving_events = detect_outlier_events(
        timeseries_container=timeseries_container, variable="calving", stationary_check=False
    )

    return plot_summary_timeseries(
        timeseries_container.timeseries["calving"],
        outliers=(calving_events[4], calving_events[2]),
        margin={"t": 0, "l": 150, "b": 25, "r": 50},
        y_label="Calving (Gt/year)",
    )


@callback(
    Output("insights_calving_detail_div", "style"),
    Output("calving_graph_div", "style"),
    Input("insights_btn_show_calving", "n_clicks"),
)
def button_detailed_calving(n_clicks):  # noqa: ANN001, ANN201, D103
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}


@callback(
    Output("insights_headline_title_sg_lakes_div", "children"),
    Output("insights_headline_text_sg_lakes_div", "children"),
    Output("insights_btn_show_sg_lakes_div", "style"),
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_sg_lake_headline(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf == "thwaites":
        lakes_events, _, _ = await example_subglacial_lake_analysis(
            ice_shelf,
            datetime(year_range[0], 1, 1),
            datetime(year_range[1], 1, 1),
        )

        if len(lakes_events) > 0:
            if len(lakes_events) > 1:
                event_s = "events"
                are_is = "are"
            else:
                event_s = "event"
                are_is = "is"

            return (
                html.Div(html.P(html.Strong("SUBGLACIAL LAKES:"), className="insights_headline_text")),
                html.Div(
                    html.P(
                        (
                            f"There {are_is} {num2words(len(lakes_events))} subglacial lakes with large volume changes "
                            f"{event_s} in the {year_range[1] - year_range[0]} year period chosen."
                        ),
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
    Output("sg_lakes_graph_vol_change", "figure"),
    Output("sg_lakes_graph_rate_vol_change", "figure"),
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def plot_sg_lake(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    ice_shelf = "thwaites"

    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    _, _, timeseries_container = await example_subglacial_lake_analysis(
        ice_shelf,
        datetime(year_range[0], 1, 1),
        datetime(year_range[1], 1, 1),
    )

    var = "volume_change"

    variables = [
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW70-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW124-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW142-" + var,
        "SUGLACIAL_LAKE_SUBGLACIAL_LAKES_THW170-" + var,
    ]
    fig_vol_change = plot_summary_timeseries_multiple(
        timeseries_container,
        variables=variables,
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
    fig_rate_vol_change = plot_summary_timeseries_multiple(
        timeseries_container,
        variables=variables,
        outliers=None,
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
def button_detailed_sg_lake(n_clicks):  # noqa: ANN001, ANN201, D103
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}, {"display": "none"}


@callback(
    Output("insights_headline_title_basal_div", "children"),
    Output("insights_headline_text_basal_div", "children"),
    Output("insights_btn_show_basal_div", "style"),
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_basal_headline(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is not None:
        _, timeseries_container = await query_summary_data(
            ice_shelf=ice_shelf,
            start_year=year_range[0],
            end_year=year_range[1],
        )

        bm_change_brackets = dectect_changes(dataset=timeseries_container, variable="basal_melt_flux")

        if len(bm_change_brackets) > 0:
            medians = calculate_median_change_brackets(
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
                        f"There has been a {np.abs(median_change): .1f}% {mag} in basal melt flux "
                        f"in the selected {year_range[1] - year_range[0]} year period.",
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
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_basal_melt_flux_detail(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is not None:
        _, timeseries_container = await query_summary_data(
            ice_shelf=ice_shelf,
            start_year=year_range[0],
            end_year=year_range[1],
        )

        bm_change_brackets = dectect_changes(dataset=timeseries_container, variable="basal_melt_flux")

        if len(bm_change_brackets) == 1:
            main = html.P(
                "There have been no significant changes in basal melt flux detected in the selcted period.",
                className="insights_calving_summary_text",
            )
        else:
            last_change = timeseries_container.timeseries["basal_melt_flux"].measurements.index[bm_change_brackets[-2]]
            main = html.P(
                (
                    f"There have been {num2words(len(bm_change_brackets) - 1)} "
                    "periods where the basal melt flux has changed in the selected time span "
                    f"over {ice_shelf} Ice shelf. The most recent was in {last_change}."
                ),
                className="insights_calving_summary_text",
            )

        text = [
            main,
        ]
        return html.Div(text)
    return html.Div()


@callback(
    Output("basal_graph", "figure"),
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def plot_basal_melt_flux(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf is None:
        ice_shelf = "thwaites"

    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    basal_events = detect_outlier_events(
        timeseries_container=timeseries_container, variable="basal_melt_flux", stationary_check=False
    )

    bm_change_brackets = dectect_changes(dataset=timeseries_container, variable="basal_melt_flux")

    return plot_summary_timeseries(
        timeseries_container.timeseries["basal_melt_flux"],
        outliers=(basal_events[4], basal_events[2]),
        brackets=bm_change_brackets,
        margin={"t": 0, "l": 150, "b": 25, "r": 50},
        y_label="Basal Melt Flux (Gt/year)",
    )


@callback(
    Output("insights_basal_detail_div", "style"),
    Output("basal_graph_div", "style"),
    Input("insights_btn_show_basal", "n_clicks"),
)
def button_detailed_basal_melt_flux(n_clicks):  # noqa: ANN001, ANN201, D103
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}


@callback(
    Output("insights_headline_title_cov_analysis_div", "children"),
    Output("insights_headline_text_cov_analysis_div", "children"),
    Output("insights_btn_show_cov_analysis_div", "style"),
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_cov_analysis_headline(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf == "thwaites":
        strong_correlation, medium_correlation, weak_correlation = await example_covariate_analysis_thwaites(
            date_start=datetime(year_range[0], 1, 1),
            date_end=datetime(year_range[1], 1, 1),
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
    Input("ice-shelf-selection", "value"),
    Input("summary-year-selector", "value"),
)
async def insight_update_cov_analysis_detail(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    if ice_shelf == "thwaites":
        strong_correlation, medium_correlation, weak_correlation = await example_covariate_analysis_thwaites(
            date_start=datetime(year_range[0], 1, 1),
            date_end=datetime(year_range[1], 1, 1),
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
                        (
                            " ".join(" ".join(corr_vars[0][:-1].lower().capitalize().split("-")[:-1]).split("_")[:-1])
                            + f" and {' '.join(' '.join(corr_vars[1][:-1].lower().split('-')[:-1]).split('_')[:-1])} "
                            f"(correlation coefficient = {strong_correlation[corr_vars][0]: .2f}, "
                            f"p-value = {strong_correlation[corr_vars][1]: .3f})"
                        ),
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
                        (
                            " ".join(" ".join(corr_vars[0][:-1].lower().capitalize().split("-")[:-1]).split("_")[:-1])
                            + f" and {' '.join(' '.join(corr_vars[1][:-1].lower().split('-')[:-1]).split('_')[:-1])} "
                            f"(correlation coefficient = {medium_correlation[corr_vars][0]: .2f}, "
                            f"p-value = {medium_correlation[corr_vars][1]: .3f})"
                        ),
                        className="insights_calving_summary_text_no_just_indent",
                    )
                )
        else:
            medium_corr_text = html.P("", className="insights_calving_summary_text_no_just_indent")

        if len(weak_correlation) > 0:
            weak_corr_text = html.P(
                " The following weak correlations have been observed: ",
                className="insights_calving_summary_text_no_just",
            )
            for corr_vars in weak_correlation:
                weak_corr_text_ul.append(
                    html.P(
                        (
                            " ".join(" ".join(corr_vars[0][:-1].lower().capitalize().split("-")[:-1]).split("_")[:-1])
                            + f" and {' '.join(' '.join(corr_vars[1][:-1].lower().split('-')[:-1]).split('_')[:-1])} "
                            f"(correlation coefficient = {weak_correlation[corr_vars][0]: .2f}, "
                            f"p-value = {weak_correlation[corr_vars][1]: .3f})"
                        ),
                        className="insights_calving_summary_text_no_just_indent",
                    )
                )
        else:
            weak_corr_text = html.P("", className="insights_calving_summary_text_no_just_indent")

        text = [
            strong_corr_text,
            *strong_corr_text_ul,
            medium_corr_text,
            *medium_corr_text_ul,
            weak_corr_text,
            *weak_corr_text_ul,
        ]
        return html.Div(text)
    return html.Div()


@callback(
    Output("insights_cov_analysis_detail_div", "style"),
    Output("cov_analysis_graph_div", "style"),
    Input("insights_btn_show_cov_analysis", "n_clicks"),
)
def button_detailed_cov_analysis(n_clicks):  # noqa: ANN001, ANN201, D103
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}, {"text-align": "center", "display": "block"}
    return {"display": "none"}, {"text-align": "center", "display": "none"}
