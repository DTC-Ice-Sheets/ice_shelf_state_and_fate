"""Component for selecting from the available datasets and variables."""

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
    DatasetSummaryRequest,
    IceShelvesApi,
)
from dtc_query_client.helpers import iterate_job

from use_case_3.dashboard.components.error_banner import create_error_banner
from use_case_3.dashboard.components.progress_dialog import create_progress_dialog
from use_case_3.dashboard.plotting import plot_dataset_summary
from use_case_3.dashboard.query_api import (
    api_config,
    config_dict,
    s3_client,
)

logger = logging.getLogger(__name__)

REGIONS = {
    "ice_shelf": {
        "name": "Ice Shelf",
        "color": "lightsalmon",
    },
    "ocean": {
        "name": "Ocean",
        "color": "lightblue",
    },
    "climate": {
        "name": "Climate",
        "color": "lightgreen",
    },
}


def _create_variable_entry(dataset_id: str, variable_id: str) -> html.Div:
    dataset = config_dict[dataset_id]
    variable = dataset["variables"][variable_id]
    return html.Div(
        [
            html.Div(
                [
                    dbc.Checkbox(
                        id={
                            "type": "data-selector",
                            "index": f"{dataset_id}.{variable_id}",
                        },
                        label=variable["label"],
                        value=False,
                    ),
                    dbc.Button(
                        "Preview...",
                        color="link",
                        id={"type": "variable-preview-button", "index": f"{dataset_id}.{variable_id}"},
                    ),
                ],
                style={"display": "flex", "justify-content": "space-between", "align-items": "center"},
            )
        ],
        id={
            "type": "data-selector-container",
            "index": f"{dataset_id}.{variable_id}",
        },
    )


def _create_dataset_entry(dataset_id: str) -> dbc.Card:
    dataset = config_dict[dataset_id]
    return dbc.Card(
        [
            html.H5(dataset["label"]),
            *[_create_variable_entry(dataset_id, variable_id) for variable_id in config_dict[dataset_id]["variables"]],
        ],
        id={"type": "data-selector-group", "index": dataset_id},
        body=True,
        color=REGIONS[dataset["region"]]["color"],
    )


def create_dataset_selection() -> html.Div:
    """Create a dataset selection component with search functionality.

    Returns
    -------
    html.Div
        A div containing the dataset selection component.
    """
    return html.Div(
        [
            create_error_banner("dataset-preview-error-message"),
            create_progress_dialog(
                title="Generating Dataset Preview",
                index="dataset-preview-generator",
                cancellable=True,
            ),
            dbc.Modal(
                dcc.Graph(id="dataset-preview-graph"),
                id="dataset-preview",
            ),
            html.H4("Select Datasets", style={"text-align": "center"}),
            dbc.Input(
                id="dataset-search-input",
                placeholder="Search datasets",
                type="search",
            ),
            html.Div(
                [_create_dataset_entry(dataset_id) for dataset_id in config_dict],
                style={
                    "display": "flex",
                    "flex-direction": "column",
                    "gap": "1em",
                    "max-height": "80vh",
                    "overflow-y": "auto",
                },
            ),
        ],
    )


@callback(
    Output({"type": "data-selector-group", "index": ALL}, "style"),
    Output({"type": "data-selector-container", "index": ALL}, "style"),
    Input("dataset-search-input", "value"),
    State({"type": "data-selector-group", "index": ALL}, "id"),
    State({"type": "data-selector", "index": ALL}, "id"),
    State({"type": "data-selector", "index": ALL}, "value"),
)
def filter_datasets(
    search_value: str,
    dataset_ids: list[dict[str, str]],
    variable_ids: list[dict[str, str]],
    variables_selected: list[bool],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Apply search filtering to visibility of datasets and variables."""
    if not search_value:
        return (
            [{"display": "block"} for _ in dataset_ids],
            [{"display": "block"} for _ in variable_ids],
        )

    variable_styles = []
    seen_datasets = set()
    for variable_id, selected in zip(variable_ids, variables_selected, strict=True):
        if selected or search_value.lower() in variable_id["index"].lower():
            variable_styles.append({"display": "block"})
            seen_datasets.add(variable_id["index"].split(".", maxsplit=1)[0])
        else:
            variable_styles.append({"display": "none"})
    dataset_styles = [
        {"display": "block"} if dataset_id["index"] in seen_datasets else {"display": "none"}
        for dataset_id in dataset_ids
    ]
    return dataset_styles, variable_styles


@callback(
    Output("dataset-preview", "is_open"),
    Output("dataset-preview-graph", "figure"),
    Output("dataset-preview-error-message", "is_open"),
    Output("dataset-preview-error-message", "children"),
    Input({"type": "variable-preview-button", "index": ALL}, "n_clicks"),
    State("ice-shelf-selection", "value"),
    State("summary-year-selector", "value"),
    running=[
        (Output({"type": "progress-dialog-modal", "index": "dataset-preview-generator"}, "is_open"), True, False),
        (Output({"type": "progress-dialog-state", "index": "dataset-preview-generator"}, "data"), "", ""),
        (
            Output({"type": "progress-dialog-status-message", "index": "dataset-preview-generator"}, "children"),
            "Setting up...",
            "Done!",
        ),
        (Output({"type": "progress-dialog-progress-bar", "index": "dataset-preview-generator"}, "value"), 0, 100),
    ],
    progress=[
        Output({"type": "progress-dialog-status-message", "index": "dataset-preview-generator"}, "children"),
        Output({"type": "progress-dialog-progress-bar", "index": "dataset-preview-generator"}, "value"),
    ],
    background=True,
    prevent_initial_call=True,
)
async def generate_dataset_preview(
    set_progress: Callable[[tuple[str, int]], None],
    _n_clicks: int,
    ice_shelf: str,
    year_range: tuple[int, int],
) -> tuple[bool, go.Figure | None, bool, str]:
    """Open the dataset preview generator dialog when a preview button is clicked."""
    dataset_id, variable_id = dash.ctx.triggered_id["index"].split(".", maxsplit=1)
    try:
        async with ApiClient(api_config) as client:
            res = await IceShelvesApi(client).run_dataset_summary(
                ice_shelf_id=ice_shelf,
                dataset_summary_request=DatasetSummaryRequest(
                    start_time=datetime(year=year_range[0], month=1, day=1),
                    end_time=datetime(year=year_range[1], month=1, day=1),
                    name=(
                        f"{config_dict[dataset_id]['label']} "
                        f"{config_dict[dataset_id]['variables'][variable_id]['label']}"
                    ),
                    dataset=dataset_id,
                    variable=variable_id,
                    extent=config_dict[dataset_id]["region"],
                ),
            )
            job_id = res.job_id
            set_props({"type": "progress-dialog-state", "index": "dataset-preview-generator"}, {"data": job_id})
            set_progress(("Starting...", 6))

            async for res in iterate_job(client, job_id):
                pdone = res.progress_done
                ptotal = res.progress_total
                percent = 5 if ptotal == 0 else max(5, min(95, int(pdone / ptotal * 100)))
                set_progress((res.status + "...", percent))

            set_progress(("Downloading result...", 90))

            output_url = res.outputs["dataset-summary-module"]["output_json"]
            bucket, key = output_url.removeprefix("s3://").removeprefix("S3://").split("/", 1)
            b = BytesIO()
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=b)
            b.seek(0)
            plot_data = json.load(b)

            set_progress(("Plotting...", 95))
            fig = plot_dataset_summary(plot_data)
    except Exception as e:
        logger.exception("Error occurred while generating plot")
        if "res" in locals():
            logger.error(res)
        return False, None, True, str(e)

    return True, fig, False, ""
