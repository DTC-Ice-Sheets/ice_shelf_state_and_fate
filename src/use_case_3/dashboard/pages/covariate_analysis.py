"""Perform covariate analysis on selected variables."""

import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Callable

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html, set_props
from dtc_query_client import (
    ApiClient,
    CovariateAnalysisRequest,
    CovariateAnalysisType,
    IceShelvesApi,
    JobsApi,
    VariableItem,
)
from dtc_query_client.helpers import iterate_job

from use_case_3.dashboard.components.collapsible_group import create_collapsible_group
from use_case_3.dashboard.components.ice_shelf_selection import create_ice_shelf_map, create_ice_shelf_selection
from use_case_3.dashboard.dashboard_layout import layout_navbar
from use_case_3.dashboard.plotting import plot_covariate_analysis
from use_case_3.dashboard.query_api import (
    api_config,
    config_dict,
    s3_client,
)

dash.register_page(__name__)

logger = logging.getLogger(__name__)

navbar = layout_navbar(page_title="Covariate Analysis")


def _create_dataset_selector(dataset_id: str) -> html.Div:
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
                                "This is the analysis module of the digital twin. "
                                "You can use the tools provided to conduct detailed analyses on the Antarctic "
                                "ice shelves of your choice using both earth observation, in-situ and reanalysis "
                                "datasets. Select from the ice, ocean and climate datasets provided to discover "
                                "significant variables and interactions between variables which could have "
                                "contributed to the current state of the ice shelves.",
                                className="analysis_intro",
                            ),
                            html.P(
                                "If you prefer to conduct the analysis offline, you can select the datasets of "
                                "interest and use the download function to retrieve them within the specified "
                                "spatial and temporal extents.",
                                className="analysis_intro",
                            ),
                            html.Div(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="select-analysis-type",
                                                value="corr",
                                                options=[
                                                    {"label": "Correlation", "value": "corr"},
                                                    {"label": "Cross-correlation", "value": "cross_corr"},
                                                    {"label": "Granger Causality", "value": "causal"},
                                                ],
                                            ),
                                            dbc.Button("Run Analysis", id="run_analysis-button", n_clicks=0),
                                        ]
                                    ),
                                    *create_ice_shelf_selection(),
                                    create_ice_shelf_map(),
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
                                                _create_dataset_selector(dataset_id)
                                                for dataset_id in [
                                                    key
                                                    for key, val in config_dict.items()
                                                    if val["region"] == "ice_shelf"
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
                                                _create_dataset_selector(dataset_id)
                                                for dataset_id in [
                                                    key for key, val in config_dict.items() if val["region"] == "ocean"
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
                                            *[
                                                _create_dataset_selector(dataset_id)
                                                for dataset_id in [
                                                    key for key, val in config_dict.items() if val["region"] == "both"
                                                ]
                                            ],
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
                        [dcc.Graph(id="results-plot")],
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
    Output("results-plot", "figure"),
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
async def conduct_covariate_analysis_integrated(
    set_progress: Callable[[tuple[str, int]], None],
    n_clicks: int,
    ice_shelf: str,
    year_range: tuple[int, int],
    analysis_type: str,
    selector_ids: list[str],
    selector_values: list[str],
) -> tuple[go.Figure | None, bool, str]:
    """Submit a job to the backend to conduct the covariate analysis.

    Updates a progress bar while the job is running.
    Provides an output image URL when done.
    """
    # Extract tuple of (dataset_id, variable_id) from the html ID of each selected variable
    ids: list[list[str]] = [
        id_["index"].split(".", maxsplit=1) for id_, value in zip(selector_ids, selector_values, strict=True) if value
    ]

    variables = [
        VariableItem(
            name=f"{config_dict[dataset_id]['label']} {config_dict[dataset_id]['variables'][variable_id]['label']}",
            dataset=dataset_id,
            variable=variable_id,
            extent=config_dict[dataset_id]["region"],
        )
        for dataset_id, variable_id in ids
    ]

    try:
        async with ApiClient(api_config) as client:
            res = await IceShelvesApi(client).run_covariate_analysis(
                ice_shelf_id=ice_shelf,
                covariate_analysis_request=CovariateAnalysisRequest(
                    start_time=datetime(year=year_range[0], month=1, day=1),
                    end_time=datetime(year=year_range[1], month=1, day=1),
                    variables=variables,
                    analysis_type=CovariateAnalysisType(analysis_type),
                ),
            )
            job_id = res.job_id
            set_props("job-id", {"data": job_id})

            try:
                async for res in iterate_job(client, job_id):
                    pdone = res.progress_done
                    ptotal = res.progress_total
                    percent = 5 if ptotal == 0 else max(5, min(95, int(pdone / ptotal * 100)))
                    set_progress(
                        (
                            res.status + "...",
                            percent,
                        )
                    )
            except RuntimeError as e:
                return None, True, str(e)

            set_progress(("Downloading result...", 95))

            output_url = res.outputs["use-case-3-covariate-analyser"]["output_json"]
            bucket, key = output_url.lower().removeprefix("s3://").split("/", 1)
            b = BytesIO()
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=b)
            b.seek(0)
            plot_data = json.load(b)
            fig = plot_covariate_analysis(plot_data)
    except Exception as e:
        logger.exception("Error occurred while generating plot")
        if "res" in locals():
            logger.error(res)
        return None, True, str(e)
    return fig, False, ""


@callback(
    Output("cancel-button", "disabled"),
    Output("error-message", "is_open", allow_duplicate=True),
    Output("error-message", "children", allow_duplicate=True),
    Input("cancel-button", "n_clicks"),
    State("job-id", "data"),
    prevent_initial_call=True,
)
async def cancel_job(n_clicks: int, job_id: str) -> tuple[bool, bool, str]:
    """Issues a request to cancel the running job."""
    if not job_id:
        return True, False, ""
    if not n_clicks:
        return False, False, ""
    try:
        async with ApiClient(api_config) as client:
            await JobsApi(client).cancel_job(job_id=job_id)
    except Exception as e:  # noqa: BLE001
        logger.error(e)
        return False, True, str(e)
    return True, False, ""
