import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

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

ice_shelf_inital = None

navbar = layout_navbar(page_title="Insights")

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
                            "Welcome to Digital Twin Ice Sheets. Choose an ice sheet below to uncover key insights relating to the state and the fate of this ice shelf.",
                            className="insights_intro_text",
                        ),
                        html.Div(
                            dbc.Select(
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
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id="ice_shelf_map",
                            style={"text-align": "center", "border": "0px"},
                        ),
                    ],
                    width=4,
                ),
            ]
        ),
    ]
)
