"""Perform covariate analysis on selected variables."""

import json
import logging
from datetime import datetime
from io import BytesIO
from shutil import make_archive, rmtree
from typing import Callable

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html, set_props
from dtc_query_client import (
    ApiClient,
    CovariateAnalysisRequest,
    CovariateAnalysisType,
    JobsApi,
    StateAndFateApi,
    VariableItem,
)
from dtc_query_client.helpers import iterate_job
from dtc_wrapper.helpers import download_directory_from_s3, parse_s3_url

from use_case_3.dashboard.components.dataset_selection import create_dataset_selection
from use_case_3.dashboard.components.error_banner import create_error_banner
from use_case_3.dashboard.components.ice_shelf_selection import create_ice_shelf_selection
from use_case_3.dashboard.components.progress_dialog import create_progress_dialog
from use_case_3.dashboard.components.s3_directory_download_button import create_s3_directory_download_button
from use_case_3.dashboard.dashboard_layout import layout_navbar
from use_case_3.dashboard.plotting import plot_covariate_analysis, plot_placeholder, text_covariate_analysis
from use_case_3.dashboard.query_api import (
    api_config,
    config_dict,
    output_path,
    s3_client,
)

dash.register_page(__name__)

logger = logging.getLogger(__name__)

navbar = layout_navbar("/covariate-analysis")


layout = [
    navbar,
    dbc.Row(
        [
            # Column for the analysis parameters
            dbc.Col(
                [
                    create_error_banner(id="job-runner-error-message"),
                    create_progress_dialog(
                        title="Running Workflow",
                        index="job-runner",
                        cancellable=True,
                    ),
                    html.H1("Perform Covariate Analysis", style={"text-align": "center"}),
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
                    dbc.Form(
                        [
                            *create_ice_shelf_selection(),
                            create_dataset_selection(),
                            dbc.Row(
                                [
                                    dbc.Label("Analysis Type:", html_for="select-analysis-type", width=3),
                                    dbc.Col(
                                        dbc.Select(
                                            id="select-analysis-type",
                                            value="corr",
                                            options=[
                                                {"label": "Correlation", "value": "corr"},
                                                {"label": "Cross-correlation", "value": "cross_corr"},
                                                {"label": "Granger Causality", "value": "causal"},
                                            ],
                                        )
                                    ),
                                ]
                            ),
                            dbc.Row(
                                dbc.Button("Run Analysis", id="run-analysis-button", size="lg", color="primary"),
                            ),
                        ],
                        id="analysis_parameters",
                        style={"display": "flex", "flex-direction": "column", "gap": "0.5em"},
                    ),
                ],
                width=4,
            ),
            # Column for the results
            dbc.Col(
                [
                    dcc.Loading(
                        [
                            html.Div(
                                style={"padding-top": "25px"},
                                id="analysis_results_div",
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="results-plot", figure=plot_placeholder()),
                                    create_s3_directory_download_button(
                                        "Download Raw Data",
                                        index="download-raw-data",
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flex-direction": "column",
                                    "align-items": "center",
                                },
                            ),
                        ],
                        id="loading",
                        type="default",
                    ),
                ],
                width=6,
            ),
        ],
        style={"margin": "0"},
    ),
]


@callback(
    Output("results-plot", "figure"),
    Output("analysis_results_div", "children"),
    Output({"type": "s3-download-url", "index": "download-raw-data"}, "data"),
    Output("job-runner-error-message", "is_open"),
    Output("job-runner-error-message", "children"),
    Input("run-analysis-button", "n_clicks"),
    State("ice-shelf-selection", "value"),
    State("summary-year-selector", "value"),
    State("select-analysis-type", "value"),
    State({"type": "data-selector", "index": ALL}, "id"),
    State({"type": "data-selector", "index": ALL}, "value"),
    running=[
        (Output("run-analysis-button", "disabled"), True, False),
        (Output({"type": "progress-dialog-modal", "index": "job-runner"}, "is_open"), True, False),
        (Output({"type": "progress-dialog-state", "index": "job-runner"}, "data"), "", ""),
        (
            Output({"type": "progress-dialog-status-message", "index": "job-runner"}, "children"),
            "Setting up...",
            "Done!",
        ),
        (Output({"type": "progress-dialog-progress-bar", "index": "job-runner"}, "value"), 0, 100),
    ],
    progress=[
        Output({"type": "progress-dialog-status-message", "index": "job-runner"}, "children"),
        Output({"type": "progress-dialog-progress-bar", "index": "job-runner"}, "value"),
    ],
    background=True,
    prevent_initial_call=True,
)
async def conduct_covariate_analysis_integrated(
    set_progress: Callable[[tuple[str, int]], None],
    _n_clicks: int,
    ice_shelf: str,
    year_range: tuple[int, int],
    analysis_type: str,
    selector_ids: list[str],
    selector_values: list[str],
) -> tuple[go.Figure | None, str | None, bool, str]:
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
            name=(
                f"{config_dict[dataset_id]['label']}: "
                f"{config_dict[dataset_id]['variables'][variable_id]['label']} "
                f"[{config_dict[dataset_id]['variables'][variable_id]['units']}]"
            ),
            dataset=dataset_id,
            variable=variable_id,
            extent=config_dict[dataset_id]["region"],
        )
        for dataset_id, variable_id in ids
    ]

    try:
        async with ApiClient(api_config) as client:
            res = await StateAndFateApi(client).run_covariate_analysis(
                ice_shelf_id=ice_shelf,
                covariate_analysis_request=CovariateAnalysisRequest(
                    start_time=datetime(year=year_range[0], month=1, day=1),
                    end_time=datetime(year=year_range[1], month=1, day=1),
                    variables=variables,
                    analysis_type=CovariateAnalysisType(analysis_type),
                ),
            )
            job_id = res.job_id
            set_props({"type": "progress-dialog-state", "index": "job-runner"}, {"data": job_id})
            set_progress(("Starting...", 6))

            async for res in iterate_job(client, job_id):
                pdone = res.progress_done
                ptotal = res.progress_total
                percent = 5 if ptotal == 0 else max(5, min(95, int(pdone / ptotal * 100)))
                set_progress((res.status + "...", percent))

            set_progress(("Downloading result...", 95))

            raw_data_url = res.outputs["use-case-3-timeseries-preprocessor"]["output_timeseries"]
            output_url = res.outputs["use-case-3-covariate-analyser"]["output_json"]
            bucket, key = output_url.removeprefix("s3://").removeprefix("S3://").split("/", 1)
            b = BytesIO()
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=b)
            b.seek(0)
            plot_data = json.load(b)
            fig = plot_covariate_analysis(plot_data)
            summary_div = text_covariate_analysis(plot_data)
    except Exception as e:
        logger.exception("Error occurred while generating plot")
        if "res" in locals():
            logger.error(res)
        return None, None, None, True, str(e)
    return fig, summary_div, raw_data_url, False, ""


@callback(
    Output({"type": "progress-dialog-cancel-button", "index": "job-runner"}, "disabled"),
    Output({"type": "progress-dialog-cancel-error-message", "index": "job-runner"}, "is_open"),
    Output({"type": "progress-dialog-cancel-error-message", "index": "job-runner"}, "children"),
    Input({"type": "progress-dialog-cancel-button", "index": "job-runner"}, "n_clicks"),
    Input({"type": "progress-dialog-state", "index": "job-runner"}, "data"),
    prevent_initial_call=True,
)
async def cancel_job(_n_clicks: int, job_id: str) -> tuple[bool, bool, str]:
    """Issues a request to cancel the running job."""
    button_id = {"type": "progress-dialog-cancel-button", "index": "job-runner"}
    was_clicked = dash.ctx.triggered_id == button_id

    if not was_clicked:
        # New job ID. If its empty disable button, otherwise enable it.
        return job_id == "", False, ""

    if job_id == "":
        # Shouldn't be possible to click cancel if no job is running, but just ignore it
        return True, False, ""

    # Button was clicked, and we have a job ID to cancel. Issue cancel request and disable button.
    # Issue a set_props in advance to give immediate feedback in the UI.
    set_props(button_id, {"disabled": True})
    try:
        async with ApiClient(api_config) as client:
            await JobsApi(client).cancel_job(job_id=job_id)
    except Exception as e:  # noqa: BLE001
        logger.error(e)
        return False, True, str(e)
    return True, False, ""


@callback(
    Output({"type": "s3-download-download", "index": "download-raw-data"}, "data"),
    Output({"type": "s3-download-error-message", "index": "download-raw-data"}, "is_open"),
    Output({"type": "s3-download-error-message", "index": "download-raw-data"}, "children"),
    Input({"type": "s3-download-button", "index": "download-raw-data"}, "n_clicks"),
    State({"type": "s3-download-url", "index": "download-raw-data"}, "data"),
    running=[
        (Output({"type": "s3-download-button", "index": "download-raw-data"}, "disabled"), True, False),
        (Output({"type": "progress-dialog-modal", "index": "download-raw-data"}, "is_open"), True, False),
        (Output({"type": "progress-dialog-state", "index": "download-raw-data"}, "data"), "", ""),
        (
            Output({"type": "progress-dialog-status-message", "index": "download-raw-data"}, "children"),
            "Setting up...",
            "Done!",
        ),
        (Output({"type": "progress-dialog-progress-bar", "index": "download-raw-data"}, "value"), 0, 100),
    ],
    progress=[
        Output({"type": "progress-dialog-status-message", "index": "download-raw-data"}, "children"),
        Output({"type": "progress-dialog-progress-bar", "index": "download-raw-data"}, "value"),
    ],
    background=True,
    prevent_initial_call=True,
)
def s3_directory_download(
    set_progress: Callable[[tuple[str, int]], None], _n_clicks: int, s3_url: str
) -> tuple[dict | None, bool, str]:
    """Trigger a download of the raw data.

    This should absolutely be a pattern matching callback inside `s3_directory_download_button.py`.
    Unfortunately, dash throws an error that pattern matching is not supported in background callbacks.
    """
    try:
        bucket_name, s3_prefix = parse_s3_url(s3_url)
        set_progress(("Downloading data...", 33))
        local_path = download_directory_from_s3(
            bucket_name,
            s3_prefix,
            output_path,
            s3_client,
        )
        set_progress(("Compressing data...", 66))
        archive_path = make_archive(
            base_dir=local_path,
            root_dir=local_path,
            base_name=str(local_path),
            format="zip",
        )
        set_progress(("Cleaning up...", 90))
        rmtree(local_path)
    except Exception as e:
        logger.exception("Failed to prepare S3 directory download.")
        return None, True, str(e)

    return dcc.express.send_file(archive_path), False, ""
