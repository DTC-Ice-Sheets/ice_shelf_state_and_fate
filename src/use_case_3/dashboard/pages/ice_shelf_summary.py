"""Shows general statistics for a selected ice shelf."""

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, dcc, html

from use_case_3.dashboard.components.ice_shelf_selection import create_ice_shelf_map, create_ice_shelf_selection
from use_case_3.dashboard.dashboard_layout import layout_navbar
from use_case_3.dashboard.plotting import plot_gauge_indicator_timeseries, plot_summary_timeseries
from use_case_3.dashboard.process_data import generate_statistic, generate_statistic_cummulative_difference
from use_case_3.dashboard.query_api import (
    query_summary_data,
)

dash.register_page(__name__)

navbar = layout_navbar(page_title="Ice Shelf Summary")

layout = html.Div(
    [
        navbar,
        # Ice Shelf summary content
        dbc.Row(
            [
                # This column is for the summary gauges, as well as the map / selectors
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                # This column is for the initial summary gauges
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_1_figure"),
                                                    dcc.Store(data="mass_balance", id="gauge_1_store"),
                                                    # dcc.Store(data="mass_trend", id="gauge_1_store"),
                                                ],
                                                style={
                                                    "padding-top": "10px",
                                                    "margin": "0px",
                                                    "border": "0px black solid",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div([]),
                                            style={
                                                "padding": 0,
                                                "margin": 0,
                                                "border": "0px black solid",
                                                "font-size": "0.75rem",
                                                "text-align": "center",
                                                "font-weight": "bold",
                                            },
                                            id="total_mass_change_gauge_text",
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_2_figure"),
                                                    dcc.Store(data="basal_melt_flux", id="gauge_2_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div([]),
                                            style={
                                                "padding": 0,
                                                "margin": 0,
                                                "border": "0px black solid",
                                                "font-size": "0.75rem",
                                                "text-align": "center",
                                                "font-weight": "bold",
                                            },
                                            id="total_basal_melt_flux_gauge_text",
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_3_figure"),
                                                    dcc.Store(data="calving", id="gauge_3_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div([]),
                                            style={
                                                "padding": 0,
                                                "margin": 0,
                                                "border": "0px black solid",
                                                "font-size": "0.75rem",
                                                "text-align": "center",
                                                "font-weight": "bold",
                                            },
                                            id="total_calving_gauge_text",
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_4_figure"),
                                                    dcc.Store(data="discharge", id="gauge_4_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                html.Div([]),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                                id="total_discharge_gauge_text",
                                            ),
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_5_figure"),
                                                    dcc.Store(data="surface_mass_balance", id="gauge_5_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div([]),
                                            style={
                                                "padding": 0,
                                                "margin": 0,
                                                "border": "0px black solid",
                                                "font-size": "0.75rem",
                                                "text-align": "center",
                                                "font-weight": "bold",
                                            },
                                            id="total_surface_mass_balance_gauge_text",
                                        ),
                                    ],
                                    width=4,
                                ),
                                # This column is for the map, ice shelf selector and time range selector.
                                dbc.Col(
                                    [
                                        *create_ice_shelf_selection(),
                                        create_ice_shelf_map(),
                                    ],
                                    width=8,
                                ),
                            ]
                        ),
                    ]
                ),
                # This column is for the time series plots
                dbc.Col(
                    [
                        html.Div(
                            html.Div(),
                            className="summary-text",
                            id="iceshelf-summary-text-div",
                        ),
                        # Add switches for timeseries selection
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            dbc.Switch(
                                                id="switch-cumulative-mass-change",
                                                label="Cumulative mass change",
                                                value=False,
                                            ),
                                            dbc.Switch(
                                                id="switch-cumulative-thinning",
                                                label="Cumulative thinning",
                                                value=False,
                                            ),
                                            dbc.Switch(
                                                id="switch-cumulative-discharge-change",
                                                label="Cumulative discharge change",
                                                value=False,
                                            ),
                                            dbc.Switch(
                                                id="switch-cumulative-SBM-anomolies",
                                                label="Cumulative SBM anomolies",
                                                value=False,
                                            ),
                                        ],
                                    )
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            dbc.Switch(
                                                id="switch-cumulative-retreat",
                                                label="Cumulative retreat",
                                                value=False,
                                            ),
                                            dbc.Switch(
                                                id="switch-mass-balance",
                                                label="Mass balance",
                                                value=False,
                                            ),
                                            dbc.Switch(
                                                id="switch-calving-rate",
                                                label="Calving rate",
                                                value=False,
                                            ),
                                        ]
                                    )
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            dbc.Switch(
                                                id="switch-basal-melt-flux",
                                                label="Basal melt flux",
                                                value=False,
                                            ),
                                            dbc.Switch(
                                                id="switch-smb",
                                                label="SMB",
                                                value=False,
                                            ),
                                            dbc.Switch(
                                                id="switch-discharge-rate",
                                                label="Discharge rate",
                                                value=False,
                                            ),
                                        ]
                                    )
                                ),
                            ]
                        ),
                        # Row for cummulative:
                        dbc.Row(
                            [
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Cumulative mass change (Gt)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_1_figure"),
                                                    dcc.Store(data="cumulative_mass_change", id="indicator_1_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_1_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Cumulative discharge change (Gt)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_2_figure"),
                                                    dcc.Store(
                                                        data="cumulative_discharge_change", id="indicator_2_store"
                                                    ),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_2_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Cumulative retreat (Gt)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_3_figure"),
                                                    dcc.Store(data="cumulative_retreat", id="indicator_3_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_3_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Cumulative thinning (Gt)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_4_figure"),
                                                    dcc.Store(data="cumulative_thinning", id="indicator_4_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_4_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Cumulative SBM anomolies (Gt)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_5_figure"),
                                                    dcc.Store(
                                                        data="cumulative_surface_mass_balance_anomalies",
                                                        id="indicator_5_store",
                                                    ),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_5_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Mass balance (Gt yr-1)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_1_figure_deriv"),
                                                    dcc.Store(data="mass_balance", id="indicator_1_store_deriv"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_6_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Calving (Gt yr-1)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_2_figure_deriv"),
                                                    dcc.Store(data="calving", id="indicator_2_store_deriv"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_7_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("SMB (Gt yr-1)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_3_figure_deriv"),
                                                    dcc.Store(
                                                        data="surface_mass_balance", id="indicator_3_store_deriv"
                                                    ),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_8_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Basal Melt Flux (Gt yr-1)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_4_figure_deriv"),
                                                    dcc.Store(data="basal_melt_flux", id="indicator_4_store_deriv"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_9_div",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                html.P("Discharge (Gt yr-1)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-decoration": "underline",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="indicator_5_figure_deriv"),
                                                    dcc.Store(
                                                        data="discharge",
                                                        id="indicator_5_store_deriv",
                                                    ),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                    ],
                                    id="figure_10_div",
                                ),
                            ],
                        ),
                    ]
                ),
            ]
        ),
    ]
)


@callback(
    Output("iceshelf-summary-text-div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
async def update_ice_shelf_summary(ice_shelf, year_range):  # noqa: ANN001, ANN201, D103
    summary_data, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    total_mass_loss, _ = generate_statistic(timeseries_container=timeseries_container, variable="mass_balance")

    cummulative_retreat, _ = generate_statistic_cummulative_difference(
        timeseries_container=timeseries_container,
        variable="cumulative_retreat",
        year_range=year_range,
    )

    total_basal_melt_flux, _ = generate_statistic(timeseries_container=timeseries_container, variable="basal_melt_flux")
    total_calving, _ = generate_statistic(timeseries_container=timeseries_container, variable="calving")

    total_basal_melt_contribution, _ = generate_statistic(
        timeseries_container=timeseries_container,
        variable="basal_melt_flux",
        steady_state_val=summary_data["steady_state_basal_melt"].values[0][0],
    )

    lost_gained = "lost" if total_mass_loss < 0 else "gained"
    loss_gain = "loss" if total_mass_loss < 0 else "gain"

    number_of_years = year_range[1] - year_range[0]

    return html.Div(
        [
            html.P(" "),
            html.H5(f"{' '.join(ice_shelf.split('_')).title()} Ice Shelf Summary:"),
            html.P(
                (
                    f"Overall the {' '.join(ice_shelf.split('_')).title()} ice shelf has {lost_gained} "
                    f"{int(np.round(np.abs(total_mass_loss))): .0f} Gt of ice "
                    f"between {year_range[0]} and {year_range[1]}."
                ),
                style={"padding-right": "100px", "padding-top": "10px"},
            ),
            html.Ul(
                [
                    html.Li(
                        "Basal melting has contributed approximately "
                        f"{np.abs(total_basal_melt_contribution / number_of_years): .0f} Gt/yr "
                        f"to this total {loss_gain} over this period."
                    ),
                    html.Li(
                        "Calving front retreat has contributed approximately "
                        f"{np.abs(cummulative_retreat / number_of_years):.0f} Gt/yr "
                        f"to this total {loss_gain} over this period."
                    ),
                ]
            ),
            html.P(
                "Fresh water flux to the ocean is partitioned as:",
                style={"padding-right": "100px", "padding-top": "0px"},
            ),
            html.Ul(
                [
                    html.Li(f"{np.abs(total_calving / number_of_years):.1f} Gt/yr from iceberg calving."),
                    html.Li(f"{np.abs(total_basal_melt_flux / number_of_years):.1f} Gt/yr from basal melting."),
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
def plot_summary_indicator1_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_4_div", "style"),
    Input("switch-cumulative-thinning", "value"),
)
def plot_summary_indicator2_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_2_div", "style"),
    Input("switch-cumulative-discharge-change", "value"),
)
def plot_summary_indicator3_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_5_div", "style"),
    Input("switch-cumulative-SBM-anomolies", "value"),
)
def plot_summary_indicator4_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_3_div", "style"),
    Input("switch-cumulative-retreat", "value"),
)
def plot_summary_indicator5_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_6_div", "style"),
    Input("switch-mass-balance", "value"),
)
def plot_summary_indicator6_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_7_div", "style"),
    Input("switch-calving-rate", "value"),
)
def plot_summary_indicator7_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_9_div", "style"),
    Input("switch-basal-melt-flux", "value"),
)
def plot_summary_indicator8_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_8_div", "style"),
    Input("switch-smb", "value"),
)
def plot_summary_indicator9_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("figure_10_div", "style"),
    Input("switch-discharge-rate", "value"),
)
def plot_summary_indicator10_visible(visibility):  # noqa: ANN001, ANN201, D103
    if visibility:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("indicator_1_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_1_store", "data"),
)
async def plot_summary_indicator1(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_2_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_2_store", "data"),
)
async def plot_summary_indicator2(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_3_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_3_store", "data"),
)
async def plot_summary_indicator3(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_4_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_4_store", "data"),
)
async def plot_summary_indicator4(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_5_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_5_store", "data"),
)
async def plot_summary_indicator5(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_1_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_1_store_deriv", "data"),
)
async def plot_summary_timeseries1(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_2_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_2_store_deriv", "data"),
)
async def plot_summary_timeseries2(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_3_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_3_store_deriv", "data"),
)
async def plot_summary_timeseries3(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_4_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_4_store_deriv", "data"),
)
async def plot_summary_timeseries4(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("indicator_5_figure_deriv", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("indicator_5_store_deriv", "data"),
)
async def plot_summary_timeseries5(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_summary_timeseries(timeseries_container.timeseries[variable])


@callback(
    Output("gauge_1_figure", "figure"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
    State("gauge_1_store", "data"),
)
async def plot_gauge_indicator1(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_gauge_indicator_timeseries(timeseries_container, variable)


@callback(
    Output("total_mass_change_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator1_text(year_range):  # noqa: ANN001, ANN201, D103
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
async def plot_gauge_indicator2(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_gauge_indicator_timeseries(timeseries_container, variable)


@callback(
    Output("total_basal_melt_flux_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator2_text(year_range):  # noqa: ANN001, ANN201, D103
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
async def plot_gauge_indicator3(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_gauge_indicator_timeseries(timeseries_container, variable)


@callback(
    Output("total_calving_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator3_text(year_range):  # noqa: ANN001, ANN201, D103
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
async def plot_gauge_indicator4(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_gauge_indicator_timeseries(timeseries_container, variable)


@callback(
    Output("total_discharge_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator4_text(year_range):  # noqa: ANN001, ANN201, D103
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
async def plot_gauge_indicator5(ice_shelf, year_range, variable):  # noqa: ANN001, ANN201, D103
    _, timeseries_container = await query_summary_data(
        ice_shelf=ice_shelf,
        start_year=year_range[0],
        end_year=year_range[1],
    )

    return plot_gauge_indicator_timeseries(timeseries_container, variable)


@callback(
    Output("total_surface_mass_balance_gauge_text", "children"),
    Input("summary_year_selector", "value"),
)
def plot_gauge_indicator5_text(year_range):  # noqa: ANN001, ANN201, D103
    return html.Div(
        html.P(
            [
                "Total surface mass balance between",
                html.Br(),
                f"{year_range[0]} and {year_range[1]} (Gt)",
            ]
        ),
    )
