import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from use_case_3.dashboard import query_data
from use_case_3.dashboard.dashboard_layout import (
    create_switch_button_group,
    layout_navbar,
)

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

navbar = layout_navbar(page_title="Covariate Analysis")

# switch_group_thwaites_basal_melt_anoms = create_switch_button_group()

layout = [
    navbar,
    dbc.Row(
        [
            # Column for the analysis parameters
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
                                                value="1",
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
            # Column for the datasets
            dbc.Col(
                [
                    html.Div(
                        [
                            # Container for the ice related variables
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Store(id="switch-values"),
                                            html.H6("Ice Shelf Datasets: ", className="analysis_heading6"),
                                            create_switch_button_group(dataset_id="polarice_elevation_change"),
                                            # create_switch_button_group(dataset_id="elevation_change_eolis"),
                                            create_switch_button_group(dataset_id="rema_v2"),
                                            create_switch_button_group(dataset_id="bedmachine_v3"),
                                            create_switch_button_group(dataset_id="polar_plus_ice_shelves_basal_melt"),
                                            create_switch_button_group(
                                                dataset_id="polar_plus_ice_shelves_calving_front"
                                            ),
                                            create_switch_button_group(dataset_id="sub_glacial_lakes"),
                                            create_switch_button_group(dataset_id="its_live_basal_melt"),
                                            create_switch_button_group(dataset_id="its_live_velocity"),
                                            create_switch_button_group(dataset_id="racmo"),
                                            create_switch_button_group(dataset_id="moorings_1d"),
                                            create_switch_button_group(dataset_id="subglacial_lake_discharge_1d"),
                                            create_switch_button_group(dataset_id="subglacial_lakes_volume_change_1d"),
                                            create_switch_button_group(
                                                dataset_id="basal_melt_anomalies_pine_island_1d"
                                            ),
                                            create_switch_button_group(dataset_id="basal_melt_anomalies_thwaites_1d"),
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
                                            create_switch_button_group(dataset_id="ocean_reanalysis"),
                                            create_switch_button_group(dataset_id="sea_ice_concentration"),
                                            create_switch_button_group(dataset_id="sea_ice_edge"),
                                            create_switch_button_group(dataset_id="sea_ice_thickness"),
                                            create_switch_button_group(dataset_id="ocean_moorings"),
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
                                            create_switch_button_group(dataset_id="southern_annular_mode"),
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
                            html.Div(
                                id="results-plot",
                                style={"text-align": "center"},
                            ),
                            # html.Img(src="assets/analysis_output.png", className="analysis_output"),
                        ],
                        style={"padding-top": "25px"},
                    ),
                ],
                width=5,
            ),
        ]
    ),
]
