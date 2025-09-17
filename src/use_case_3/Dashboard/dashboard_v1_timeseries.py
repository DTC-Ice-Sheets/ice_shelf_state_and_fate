
import dash_bootstrap_components as dbc
import mapping
import plotting
import query_data
from dash import Dash, Input, Output, State, callback, dcc, html

ice_shelf_inital = "Thwaites"

### Import data ###
ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj = query_data.query_ice_shelf_wide_data()
summary_data_initial, _, _ = query_data.query_summary_data(
    ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf_inital
)

# print(summary_data[variable_initial])

### Create the initial map ###
ice_shelf_map = mapping.create_ice_shelf_map(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf_inital)

### Build Dash app ###

# app = Dash()
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

structure = html.Div(
    [
        dbc.Row(
            dbc.Col(html.H1(children="Ice Shelf Summary", style={"textAlign": "center", "border": "2px black solid"}))
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_1_figure"),
                                                    dcc.Store(data="mass_trend", id="gauge_1_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                html.P(
                                                    [
                                                        "Total mass change rate between",
                                                        html.Br(),
                                                        "1997 and 2021 (Gt yr-1)",
                                                    ]
                                                ),
                                                # html.P( <br> ),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_2_figure"),
                                                    dcc.Store(data="steady_state_discharge", id="gauge_2_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                html.P(
                                                    [
                                                        "Steady-state discharge rate between",
                                                        html.Br(),
                                                        "1997 and 2021 (Gt yr-1)",
                                                    ]
                                                ),
                                                # html.P("Steady-state discharge rate between 1997 and 2021 (Gt yr-1)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_3_figure"),
                                                    dcc.Store(data="steady_state_calving", id="gauge_3_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                html.P(
                                                    [
                                                        "Steady-state calving rate between",
                                                        html.Br(),
                                                        "1997 and 2021 (Gt yr-1)",
                                                    ]
                                                ),
                                                # html.P("Steady-state calving rate between 1997 and 2021 (Gt yr-1)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_4_figure"),
                                                    dcc.Store(data="steady_state_basal_melt", id="gauge_4_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                # html.P(
                                                #     "Steady-state basal melting rate between 1997 and 2021 (Gt yr-1)"
                                                # ),
                                                html.P(
                                                    [
                                                        "Steady-state basal melting rate between",
                                                        html.Br(),
                                                        "1997 and 2021 (Gt yr-1)",
                                                    ]
                                                ),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                [
                                                    dcc.Graph(id="gauge_5_figure"),
                                                    dcc.Store(data="steady_state_smb", id="gauge_5_store"),
                                                ],
                                                style={"padding": 0, "margin": 0, "border": "0px black solid"},
                                            )
                                        ),
                                        dbc.Row(
                                            html.Div(
                                                html.P(
                                                    [
                                                        "Steady-state SMB rate between",
                                                        html.Br(),
                                                        "1997 and 2021 (Gt yr-1)",
                                                    ]
                                                ),
                                                # html.P("Steady-state SMB rate between 1997 and 2021 (Gt yr-1)"),
                                                style={
                                                    "padding": 0,
                                                    "margin": 0,
                                                    "border": "0px black solid",
                                                    "font-size": "0.75rem",
                                                    "text-align": "center",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Div(
                                            dcc.Dropdown(
                                                ice_shelf_polygons_proj["Name"].unique(),
                                                ice_shelf_inital,
                                                id="ice_shelf_selection",
                                                style={},
                                            ),
                                            style={"border": "0px black solid"},
                                        ),
                                        html.Div(
                                            dcc.RangeSlider(
                                                summary_data_initial["time"].min().values.astype(object).year,
                                                summary_data_initial["time"].max().values.astype(object).year,
                                                1,
                                                value=[
                                                    summary_data_initial["time"].min().values.astype(object).year,
                                                    summary_data_initial["time"].max().values.astype(object).year,
                                                ],
                                                id="summary_year_selector",
                                                marks=None,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                    "style": {"color": "White", "fontSize": "20px"},
                                                },
                                            ),
                                            style={"border": "0px black solid"},
                                        ),
                                        html.Div(
                                            id="ice_shelf_map",
                                            style={"text-align": "center", "border": "0px black solid"},
                                        ),
                                    ],
                                    width=8,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        # Row for cummulative:
                        dbc.Row(
                            [
                                html.Div(
                                    html.H2("Cummulative"),
                                    style={
                                        "border": "0px black solid",
                                        "text-align": "center",
                                        "text-decoration": "underline",
                                    },
                                )
                            ],
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
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
                                    width=6,
                                ),
                                dbc.Col(
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
                                    width=6,
                                ),
                            ],
                        ),
                        # Row for derivatives:
                        dbc.Row(
                            [
                                html.Div(
                                    html.H2("Derivatives"),
                                    style={
                                        "border": "0px black solid",
                                        "text-align": "center",
                                        "text-decoration": "underline",
                                    },
                                )
                            ],
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
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
                                    width=6,
                                ),
                                dbc.Col(
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
                                    width=6,
                                ),
                            ],
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(html.H1(children="Covaraite Analysis", style={"textAlign": "center", "border": "2px black solid"}))
        ),
    ]
)

# Requires Dash 2.17.0 or later
app.layout = [structure]


@callback(Output("ice_shelf_map", "children"), Input("ice_shelf_selection", "value"))
def update_graph(value):
    ice_shelf_map = mapping.create_ice_shelf_map(ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=value)

    return ice_shelf_map


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
    app.run(debug=False)
