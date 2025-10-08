import base64
import logging
from io import BytesIO
from time import sleep

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html, set_props
import pandas as pd
import geopandas as gpd
import requests

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser
from use_case_3.covariate_analyser_module.timeseries_preprocessor import TimeSeriesPreprocessor
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    JointPlotAnalysisType,
)
from use_case_3.dashboard.dashboard_layout import (
    create_switch_button_group,
    layout_navbar,
)
from use_case_3.dashboard.query_api import (
    url,
    config_dict,
    s3_client,
    query_summary_data,
    query_polygons,
    process_all_data,
)


dash.register_page(__name__)

logger = logging.getLogger(__name__)

navbar = layout_navbar(page_title="Covariate Analysis")

initial_ice_shelf = "thwaites"
initial_summary_data, _, _, _ = query_summary_data(initial_ice_shelf)

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
                                            options=query_polygons()[0].proj["Name"].unique(),
                                            value=initial_ice_shelf,
                                            id="ice_shelf_selection",
                                            style={},
                                        ),
                                        style={"border": "0px black solid", "padding-top": "10px"},
                                    ),
                                    html.Div(
                                        dcc.RangeSlider(
                                            initial_summary_data["time"].min().values.astype(object).year,
                                            initial_summary_data["time"].max().values.astype(object).year,
                                            1,
                                            value=[
                                                initial_summary_data["time"].min().values.astype(object).year,
                                                initial_summary_data["time"].max().values.astype(object).year,
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
                    dbc.Modal(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Store(id="job-id", data=""),
                                        html.H3("Running Workflow", style={"alignSelf": "center"}),
                                        html.H4("Setting up...", id="status"),
                                        dbc.Progress(
                                            id="progress",
                                            value=5,
                                            striped=True,
                                            animated=True,
                                            style={"height": "4em"},
                                        ),
                                        dbc.Button(
                                            "Cancel",
                                            id="cancel-button",
                                            color="danger",
                                            size="lg",
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "padding": "1em",
                                        "gap": "0.5em",
                                        "flexDirection": "column",
                                        "alignItems": "stretch",
                                    },
                                ),
                            ],
                        ),
                        id="job-status",
                        is_open=False,
                    ),
                    dbc.Alert(
                        "Error",
                        id="error-message",
                        color="danger",
                        is_open=False,
                        dismissable=True,
                    ),
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
                                            create_switch_button_group(dataset_id="elevation_change_eolis"),
                                            create_switch_button_group(dataset_id="rema_v2"),
                                            create_switch_button_group(dataset_id="bedmachine_v3"),
                                            create_switch_button_group(dataset_id="polar_plus_ice_shelves_basal_melt"),
                                            create_switch_button_group(
                                                dataset_id="polar_plus_ice_shelves_calving_front"
                                            ),
                                            create_switch_button_group(dataset_id="sub_glacial_lakes"),
                                            create_switch_button_group(dataset_id="its_live_basal_melt"),
                                            create_switch_button_group(dataset_id="its_live_ice_velocity"),
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
                                            create_switch_button_group(dataset_id="GLORYS12V1"),
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
                        style={"padding-top": "25px"},
                        id="analysis_results_div",
                    ),
                    dcc.Loading(
                        [html.Img(id="results-plot")],
                        id="loading",
                        type="default",
                    ),
                    # html.Div(
                    #     id="results-plot",
                    #     style={"text-align": "center"},
                    # ),
                ],
                width=5,
            ),
        ]
    ),
]


@callback(
    Output("cov_analysis_graph_div", "children"),
    Input("ice_shelf_selection", "value"),
    Input("summary_year_selector", "value"),
)
def plot_cov_analysis(ice_shelf, year_range):
    ice_shelf = "thwaites"
    ice_shelf_polygons, ocean_polygons = query_polygons()
    ice_shelf_extent = ice_shelf_polygons.epsg4326[ice_shelf_polygons.epsg4326["Name"] == ice_shelf]["geometry"].values[
        0
    ]
    ocean_extent = ocean_polygons.epsg4326[ocean_polygons.epsg4326["Name"] == ice_shelf]["geometry"].values[0]
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
        axis_labels.append(
            config_dict[variable.split("-")[0].lower()]["variables"][variable.split("-")[-1]]["y_axis_label"]
        )

    timeseries = process_all_data(variables, polygons, year_range)

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
    background=True,
    prevent_initial_call=True,
)
def conduct_covariate_analysis_integrated(
    set_progress, n_clicks, ice_shelf, year_range, analysis_type, dataset_ids, datasets
):
    ice_shelf_polygons, ocean_polygons = query_polygons()
    ice_shelf_extent = ice_shelf_polygons.epsg4326[ice_shelf_polygons.epsg4326["Name"] == ice_shelf]["geometry"].values[
        0
    ]
    ocean_extent = ocean_polygons.epsg4326[ocean_polygons.epsg4326["Name"] == ice_shelf]["geometry"].values[0]
    climate_extent = None

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
