"""Component for selecting from the available datasets and variables."""

import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Callable

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html, set_props
from dash.development.base_component import Component
from dtc_query_client import (
    ApiClient,
    DatasetSummaryRequest,
    StateAndFateApi,
)
from dtc_query_client.helpers import iterate_job

from use_case_3.dashboard.components.error_banner import create_error_banner
from use_case_3.dashboard.components.progress_dialog import create_progress_dialog
from use_case_3.dashboard.plotting import plot_dataset_summary
from use_case_3.dashboard.query_api import (
    api_config,
    config_dict,
    get_ice_shelf_geometry,
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


# def _create_dataset_selection_modal() -> dbc.Modal:
#     return dbc.Modal(
#         [
#             dbc.ModalHeader(dbc.ModalTitle("Select Datasets")),
#             dbc.ModalBody(
#                 [
#                     create_error_banner("dataset-preview-error-message"),
#                     dbc.Input(
#                         id="dataset-search-input",
#                         placeholder="Search datasets",
#                         type="search",
#                     ),
#                     html.Div(
#                         [
#                             html.Span("Region filter:"),
#                             *[
#                                 dbc.Card(
#                                     dbc.Checkbox(
#                                         id={"type": "region-filter", "index": key},
#                                         label=region["name"],
#                                         value=True,
#                                         label_style={"margin": "0"},
#                                     ),
#                                     color=region["color"],
#                                     style={"padding": "0 0.5em"},
#                                 )
#                                 for key, region in REGIONS.items()
#                             ],
#                         ],
#                         style={"display": "flex", "align-items": "center", "gap": "1em", "margin": "0.5em 0"},
#                     ),
#                     html.Div(
#                         [_create_dataset_entry(dataset_id) for dataset_id in config_dict]
#                         + [_create_dataset_entry_demo_data(dataset_id) for dataset_id in config_dict_demo_data],
#                         style={
#                             "display": "flex",
#                             "flex-direction": "column",
#                             "gap": "1em",
#                             "max-height": "75vh",
#                             "overflow-y": "auto",
#                         },
#                     ),
#                 ],
#                 style={"display": "flex", "flex-direction": "column"},
#             ),
#         ],
#         id="dataset-selection-modal",
#         size="lg",
#     )


def _create_dataset_selection_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Select Datasets")),
            dbc.ModalBody(
                [
                    create_error_banner("dataset-preview-error-message"),
                    dbc.Input(
                        id="dataset-search-input",
                        placeholder="Search datasets",
                        type="search",
                    ),
                    html.Div(
                        [
                            html.Span("Region filter:"),
                            *[
                                dbc.Card(
                                    dbc.Checkbox(
                                        id={"type": "region-filter", "index": key},
                                        label=region["name"],
                                        value=True,
                                        label_style={"margin": "0"},
                                    ),
                                    color=region["color"],
                                    style={"padding": "0 0.5em"},
                                )
                                for key, region in REGIONS.items()
                            ],
                        ],
                        style={"display": "flex", "align-items": "center", "gap": "1em", "margin": "0.5em 0"},
                    ),
                    html.Div(
                        [_create_dataset_entry(dataset_id) for dataset_id in config_dict],
                        style={
                            "display": "flex",
                            "flex-direction": "column",
                            "gap": "1em",
                            "max-height": "75vh",
                            "overflow-y": "auto",
                        },
                    ),
                ],
                style={"display": "flex", "flex-direction": "column"},
            ),
        ],
        id="dataset-selection-modal",
        size="lg",
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
            create_progress_dialog(
                title="Generating Dataset Preview",
                index="dataset-preview-generator",
                cancellable=True,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id="dataset-preview-title")),
                    dbc.ModalBody(
                        id="dataset-preview-body",
                        style={"display": "flex", "flex-direction": "column"},
                    ),
                ],
                id="dataset-preview",
                size="xl",
                backdrop="static",
            ),
            dbc.Row(
                [
                    dbc.Label("Datasets:", html_for="open-dataset-selection", width=3),
                    dbc.Col(
                        [
                            dbc.Button("Select Datasets", id="open-dataset-selection", color="primary", outline=True),
                            html.Span(
                                "0 datasets selected",
                                id="dataset-selection-count",
                                style={"margin-left": "1em"},
                            ),
                        ]
                    ),
                ]
            ),
            _create_dataset_selection_modal(),
        ],
    )


def _dataset_summary_tables(plot_data: dict) -> html.Div:
    return html.Div(
        [
            html.H3("Dimensions"),
            dbc.Table(
                [
                    html.Tr(
                        [
                            html.Th("Dimension"),
                            *[html.Td(name) for name in plot_data["dims"]],
                        ]
                    ),
                    html.Tr(
                        [
                            html.Th("Size"),
                            *[html.Td(dim["real_size"]) for dim in plot_data["dims"].values()],
                        ]
                    ),
                ]
            ),
            html.H3("Variables"),
            dbc.Table(
                [
                    html.Tr(
                        [
                            html.Th("Variable"),
                            html.Th("Dimensions"),
                        ]
                    ),
                    *[
                        html.Tr(
                            [
                                html.Td(var_name),
                                html.Td(", ".join(var_info["coords"])),
                            ]
                        )
                        for var_name, var_info in plot_data["vars"].items()
                    ],
                ]
            ),
            *(
                [dbc.Alert("Some dimensions have been downsampled for plotting purposes.", color="warning")]
                if any(dim_info["real_size"] > len(dim_info["idxs"]) for dim_info in plot_data["dims"].values())
                else []
            ),
        ],
    )


@callback(
    Output("dataset-selection-modal", "is_open"),
    Input("open-dataset-selection", "n_clicks"),
    prevent_initial_call=True,
)
def open_dataset_selection(_n_clicks: int) -> bool:
    """Open the dataset selection modal when the button is clicked."""
    return True


@callback(
    Output({"type": "data-selector-group", "index": ALL}, "style"),
    Output({"type": "data-selector-container", "index": ALL}, "style"),
    Output("dataset-selection-count", "children"),
    Input("dataset-search-input", "value"),
    Input({"type": "region-filter", "index": ALL}, "value"),
    Input({"type": "data-selector", "index": ALL}, "value"),
    State({"type": "data-selector-group", "index": ALL}, "id"),
    State({"type": "data-selector", "index": ALL}, "id"),
)
def filter_datasets(
    search_value: str,
    region_filter_values: list[bool],
    variables_selected: list[bool],
    dataset_ids: list[dict[str, str]],
    variable_ids: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    """Apply search filtering to visibility of datasets and variables."""
    selected_regions = {key for key, selected in zip(REGIONS.keys(), region_filter_values, strict=True) if selected}

    variable_styles = []
    seen_datasets = set()
    for variable_id, selected in zip(variable_ids, variables_selected, strict=True):
        dataset_id = variable_id["index"].split(".", maxsplit=1)[0]
        # Show items that are selected...
        if selected or (
            # ...and any items that pass both the search and region filter
            (not search_value or search_value.lower() in variable_id["index"].lower())
            and config_dict[dataset_id]["region"] in selected_regions
        ):
            variable_styles.append({"display": "block"})
            seen_datasets.add(dataset_id)
        else:
            variable_styles.append({"display": "none"})
    dataset_styles = [{"display": "block"} if d["index"] in seen_datasets else {"display": "none"} for d in dataset_ids]
    count = sum(variables_selected)
    count_text = f"{count} dataset{'s' if count != 1 else ''} selected"
    return dataset_styles, variable_styles, count_text


@callback(
    Output("dataset-preview", "is_open"),
    Output("dataset-preview-title", "children"),
    Output("dataset-preview-body", "children"),
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
) -> tuple[bool, str, list[Component], bool, str]:
    """Open the dataset preview generator dialog when a preview button is clicked."""
    dataset_id, variable_id = dash.ctx.triggered_id["index"].split(".", maxsplit=1)
    name = f"{config_dict[dataset_id]['label']}: {config_dict[dataset_id]['variables'][variable_id]['label']}"
    try:
        async with ApiClient(api_config) as client:
            res = await StateAndFateApi(client).run_dataset_summary(
                ice_shelf_id=ice_shelf,
                dataset_summary_request=DatasetSummaryRequest(
                    start_time=datetime(year=year_range[0], month=1, day=1),
                    end_time=datetime(year=year_range[1], month=1, day=1),
                    name=name,
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
            geom = await get_ice_shelf_geometry(ice_shelf, config_dict[dataset_id]["region"])
            fig = plot_dataset_summary(plot_data, geom)
            out = [
                _dataset_summary_tables(plot_data),
                dcc.Graph(figure=fig, style={"align-self": "center"}),
            ]
    except Exception as e:
        logger.exception("Error occurred while generating plot")
        if "res" in locals():
            logger.error(res)
        return False, "", [], True, str(e)

    return True, name, out, False, ""
