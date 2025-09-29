import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from use_case_3.dashboard import query_data
from use_case_3.dashboard.dashboard_layout import layout_navbar

dash.register_page(__name__)

# Import data ###
ice_shelf_inital = "thwaites"
# ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj = query_data.query_ice_shelf_wide_data()
ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj, ocean_polygons_epsg4326, ocean_polygons_proj = (
    query_data.query_polygons()
)
summary_data_initial, timeseries_container, _, _ = query_data.query_summary_data(
    ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf_inital
)

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
                                                    dcc.Store(data="mass_trend", id="gauge_1_store"),
                                                ],
                                                style={
                                                    "padding-top": "10px",
                                                    "margin": "0px",
                                                    "border": "0px black solid",
                                                },
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
                                # This column is for the map, ice shelf selector and time range selector.
                                dbc.Col(
                                    [
                                        html.Div(
                                            dcc.Dropdown(
                                                ice_shelf_polygons_proj["Name"].unique(),
                                                ice_shelf_inital,
                                                id="ice_shelf_selection",
                                                style={},
                                            ),
                                            style={"border": "0px black solid", "padding-top": "10px"},
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
                                            style={"border": "0px black solid", "padding-top": "10px"},
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
