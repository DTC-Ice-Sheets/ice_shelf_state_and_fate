"""Component for selecting from the available datasets and variables."""

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, html

from use_case_3.dashboard.query_api import config_dict

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
            dbc.Checkbox(
                id={
                    "type": "data-selector",
                    "index": f"{dataset_id}.{variable_id}",
                },
                label=variable["label"],
                value=False,
            )
        ]
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
    Output({"type": "data-selector", "index": ALL}, "style"),
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
