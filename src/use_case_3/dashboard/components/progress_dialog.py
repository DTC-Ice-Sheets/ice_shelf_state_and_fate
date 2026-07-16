"""Progress dialog component."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from use_case_3.dashboard.components.error_banner import create_error_banner


def create_progress_dialog(
    title: str,
    index: str,
    cancellable: bool = False,
) -> dbc.Modal:
    """Create a modal dialog for showing the progress of a long-running task.

    Parameters
    ----------
    title : str
        A static title at the top of the dialog.
    index : str
        Index used for all sub-components.
        The following component types will be created with this index:
        - progress-dialog-modal: The parent modal dialog. Set `is_open=True` to show the dialog.
        - progress-dialog-state: A dcc.Store to hold some state of the task shared between callbacks (e.g. job ID).
        - progress-dialog-status-message: Text indicating the current status of the task.
        - progress-dialog-progress-bar: A dbc.Progress bar showing the percentage completion of the task.
        - progress-dialog-cancel-button: (optional) A button to cancel the task.
        - progress-dialog-cancel-error-message: (optional) An error banner to show cancellation errors.
    cancellable : bool
        Whether to include the cancel button and error message, by default False.

    Returns
    -------
    dbc.Modal
        The progress dialog modal component (initially closed).
    """
    return dbc.Modal(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        dcc.Store(id={"type": "progress-dialog-state", "index": index}, data=""),
                        html.H3(title, style={"alignSelf": "center"}),
                        html.H4("Setting up...", id={"type": "progress-dialog-status-message", "index": index}),
                        dbc.Progress(
                            id={"type": "progress-dialog-progress-bar", "index": index},
                            value=5,
                            striped=True,
                            animated=True,
                            style={"height": "4em"},
                        ),
                        *(
                            [
                                create_error_banner({"type": "progress-dialog-cancel-error-message", "index": index}),
                                dbc.Button(
                                    "Cancel",
                                    id={"type": "progress-dialog-cancel-button", "index": index},
                                    disabled=True,
                                    color="danger",
                                    size="lg",
                                ),
                            ]
                            if cancellable
                            else []
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
        id={"type": "progress-dialog-modal", "index": index},
        is_open=False,
        backdrop="static",
    )
