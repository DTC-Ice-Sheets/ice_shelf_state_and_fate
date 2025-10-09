"""Perform covariate analysis on selected variables."""

import base64
import logging
from io import BytesIO
from time import sleep

import dash
import dash_bootstrap_components as dbc
import geopandas as gpd
import pandas as pd
import requests
from dash import ALL, Input, Output, State, callback, dcc, html, set_props

from use_case_3.dashboard.dashboard_layout import (
    create_collapsible_group,
    layout_navbar,
)
from use_case_3.dashboard.query_api import (
    config_dict,
    query_polygons,
    query_summary_data,
    s3_client,
    url,
)

dash.register_page(__name__)

logger = logging.getLogger(__name__)

navbar = layout_navbar(page_title="Covariate Analysis")

initial_ice_shelf = "thwaites"
initial_summary_data, _, _, _ = query_summary_data(initial_ice_shelf)


def create_dataset_selector(dataset_id: str) -> html.Div:
    return create_collapsible_group(
        group_id=dataset_id,
        group_label=config_dict[dataset_id]["label"],
        group_description=config_dict[dataset_id]["description"],
        content=[
            dbc.Checkbox(
                id={
                    "type": "data-selector",
                    "index": f"{dataset_id}.{var_id}",
                },
                label=var["label"],
                value=False,
            )
            for var_id, var in config_dict[dataset_id]["variables"].items()
        ],
    )


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
                                            *[
                                                create_dataset_selector(dataset_id)
                                                for dataset_id in [
                                                    "polarice_elevation_change",
                                                    "elevation_change_eolis",
                                                    "rema_v2",
                                                    "bedmachine_v3",
                                                    "polar_plus_ice_shelves_basal_melt",
                                                    "polar_plus_ice_shelves_calving_front",
                                                    "sub_glacial_lakes",
                                                    "its_live_basal_melt",
                                                    "its_live_ice_velocity",
                                                    "racmo",
                                                    "moorings_1d",
                                                    "subglacial_lake_discharge_1d",
                                                    "subglacial_lakes_volume_change_1d",
                                                    "basal_melt_anomalies_pine_island_1d",
                                                    "basal_melt_anomalies_thwaites_1d",
                                                ]
                                            ],
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
                                            *[
                                                create_dataset_selector(dataset_id)
                                                for dataset_id in [
                                                    "GLORYS12V1",
                                                    "sea_ice_concentration",
                                                    "sea_ice_edge",
                                                    "sea_ice_thickness",
                                                    "ocean_moorings",
                                                ]
                                            ],
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
                                            create_dataset_selector("southern_annular_mode"),
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
                ],
                width=5,
            ),
        ]
    ),
]


@callback(
    Output("results-plot", "src"),
    Output("error-message", "is_open", allow_duplicate=True),
    Output("error-message", "children", allow_duplicate=True),
    Input("run_analysis-button", "n_clicks"),
    State("ice_shelf_selection", "value"),
    State("summary_year_selector", "value"),
    State("select-analysis-type", "value"),
    State({"type": "data-selector", "index": ALL}, "id"),
    State({"type": "data-selector", "index": ALL}, "value"),
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
    set_progress, n_clicks, ice_shelf, year_range, analysis_type, selector_ids, selector_values
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

    # Extract tuple of (dataset_id, variable_id) from the html ID of each selected variable
    ids: list[tuple[str, str]] = [
        id_["index"].split(".") for id_, value in zip(selector_ids, selector_values, strict=True) if value
    ]

    variables = [
        {
            "name": f"{config_dict[dataset_id]['label']} {config_dict[dataset_id]['variables'][variable_id]['label']}",
            "dataset": dataset_id,
            "variable": variable_id,
            "extent": config_dict[dataset_id]["variables"][variable_id]["region"],
        }
        for dataset_id, variable_id in ids
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
