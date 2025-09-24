import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, html, dcc

from use_case_3.dashboard import query_data
from use_case_3.dashboard.dashboard_layout import layout_navbar

dash.register_page(__name__)

# Import data ###
ice_shelf_inital = "Thwaites"
ice_shelf_polygons_epsg4326, ice_shelf_polygons_proj = query_data.query_ice_shelf_wide_data()
summary_data_initial, timeseries_container, _, _ = query_data.query_summary_data(
    ice_shelf_polygons=ice_shelf_polygons_proj, ice_shelf=ice_shelf_inital
)

navbar = layout_navbar(page_title="Covariate Analysis")

layout = [
    navbar,
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        [
                            html.P(
                                "This is the analysis module of the digital twin. You can use the tools provided to conduct detailed analyses on the Antarctic ice shelves of your choice using both earth observation, in-situ and reanalysis datasets. Select from the ice, ocean and climate datasets provided to discover significant variables and interactions between variables which could have contributed to the current state of the ice shelves.",
                                className="analysis_intro",
                            ),
                            html.P(
                                "If you prefer to conduct the analysis offline, you can select the datasets of interest and use the download function to retrieve them within the specified spatial and temporal extents.",
                                className="analysis_intro",
                            ),
                            html.Div(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="select-analysis-type",
                                                value="Correlation",
                                                options=[
                                                    {"label": "Correlation", "value": "1"},
                                                    {"label": "Cross-correlation", "value": "2"},
                                                    {"label": "Granger Causality", "value": "3"},
                                                ],
                                            ),
                                            dbc.Button("Run Analysis", id="run_analysis-button", n_clicks=0),
                                        ]
                                    ),
                                    html.Div(
                                        dbc.Select(
                                            # dcc.Dropdown(
                                            options=ice_shelf_polygons_proj["Name"].unique(),
                                            value=ice_shelf_inital,
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
                                id="analysis_parameters",
                            ),
                        ],
                        style={"padding-top": "25px"},
                    ),
                ],
                width=4,
            ),
            dbc.Col(
                [
                    html.Div(
                        [
                            # Container for the ice related variables
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H6("Ice Shelf Datasets: ", className="analysis_heading6"),
                                            dbc.Switch(
                                                id="switch-elevationchange-polarice",
                                                label="Polar Ice Gridded POCA Corrected Swath Elevation:",
                                                value=False,
                                                # className="analsysis_icedata_switch",
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "This dataset contains the gridded POCA corrected swath DEMdiff data over the Amundsen Sector at a 1000m resolution.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-elevationchange-cryotempo-eolis",
                                                label="CryoTempo-EOLIS Gridded Swath Elevation Product:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "This dataset contains the gridded swath DEMdiff data over Antarctica at a 2000m resolution.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-rema",
                                                label="REMA V2 Digital Elevation Map:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the REMA V2 product, available at 100 and 1000m resolutions.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-bedmachine",
                                                label="BedMachine V3 Dataset:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the BedMachine V3 product.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-basal-melt-polar+",
                                                label="Polar+ Project Basal Melt Map:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the Polar+ Project basal melt map.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-basal-melt-its-live",
                                                label="ITSLive Basal Melt:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the ITSLive Basal Melt Dataset.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-velocity-its-live",
                                                label="ITSLive Ice Velocity:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the ITSLive Ice Velocoity Dataset.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-velocity-enveo",
                                                label="Cumulative SBM anomolies:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the ENVEO Ice Velocoity Dataset.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-subglacial-lakes",
                                                label="Subglacial Lakes Dataset:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the Subglacial Lakes Dataset.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-surface-melt-map",
                                                label="4D Antarctica Surface Melt Map:",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the 4D Antarctica Surface Melt Map.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                        ],
                                        style={"padding-left": "30px"},
                                    )
                                ],
                                id="ice_data_selector",
                            ),
                            # Container for the ocean related variables
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H6("Ocean Datasets: ", className="analysis_heading6"),
                                            dbc.Switch(
                                                id="switch-glory",
                                                label="GLORY Ocean Reanalysis:",
                                                value=False,
                                                # className="analsysis_icedata_switch",
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the GLORY Ocean Reanalysis Dataset.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                            dbc.Switch(
                                                id="switch-Ocean Moorings",
                                                label="Ocean Moorings (in situ):",
                                                value=False,
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the in situ Ocean Moorings Dataset.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                        ],
                                        style={"padding-left": "30px"},
                                    )
                                ],
                                id="ocean_data_selector",
                            ),
                            # Container for the climate related variables
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H6("Climate Datasets: ", className="analysis_heading6"),
                                            dbc.Switch(
                                                id="switch-sam",
                                                label="Southern Annular Mode Index:",
                                                value=False,
                                                # className="analsysis_icedata_switch",
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Placeholder description for the Southern Annular Mode Dataset.",
                                                        className="analysis-data-desc",
                                                    )
                                                ]
                                            ),
                                        ],
                                        style={"padding-left": "30px"},
                                    )
                                ],
                                id="climate_data_selector",
                            ),
                        ],
                        style={"padding-top": "25px"},
                    ),
                    html.Div(
                        dbc.Button("Download Data", color="primary", id="analysis-datadownload"),
                        className="d-grid gap-2",
                    ),
                ],
                width=3,
            ),
            # Column for the results
            dbc.Col(
                [
                    html.Div(
                        [
                            html.H5("Analysis Results:", className="analysis_heading5"),
                            html.P(
                                "Based on the correlation analysis conducted over Thwaites Ice Shelf, the Digital Twin has determined that the following variables have shown a significant correlation over the anlysis period, and are worthy of further investigation:",
                                className="analysis_results",
                            ),
                            html.Ul(
                                [
                                    html.Li("Total subglacial lake volume change with a correlation coefficient of X."),
                                    html.Li("Thwaites Basal Melting Anomalies with a correlation coefficient of Y."),
                                ],
                                id="analysis_output_ul",
                            ),
                            html.P(
                                "A joint plot of the main variables included in the analyis can be seen below to explore these relationshipts further:",
                                className="analysis_results",
                            ),
                            html.Img(src="assets/analysis_output.png", className="analysis_output"),
                        ],
                        style={"padding-top": "25px"},
                    ),
                ],
                width=5,
            ),
        ]
    ),
]
