"""Progress dialog component."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from use_case_3.dashboard.components.error_banner import create_error_banner


def create_progress_dialog(
    title: str,
    prefix: str,
    cancellable: bool = False,
) -> dbc.Modal:
    """Create a modal dialog for showing the progress of a long-running task.

    Parameters
    ----------
    title : str
        A static title at the top of the dialog.
    prefix : str
        A prefix to use for the IDs of the sub-components.
        The following IDs will be created:
        - {prefix}-modal: The parent modal dialog. Set `is_open=True` to show the dialog.
        - {prefix}-state: A dcc.Store to hold some state of the task shared between callbacks (e.g. job ID).
        - {prefix}-status-message: Text indicating the current status of the task.
        - {prefix}-progress-bar: A dbc.Progress bar showing the percentage completion of the task.
        - {prefix}-cancel-button: (optional) A button to cancel the task.
        - {prefix}-cancel-error-message: (optional) An error banner to show errors related to cancelling the task.
    cancellable : bool, optional
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
                        dcc.Store(id=f"{prefix}-state", data=""),
                        html.H3(title, style={"alignSelf": "center"}),
                        html.H4("Setting up...", id=f"{prefix}-status-message"),
                        dbc.Progress(
                            id=f"{prefix}-progress-bar",
                            value=5,
                            striped=True,
                            animated=True,
                            style={"height": "4em"},
                        ),
                        *(
                            [
                                create_error_banner(id=f"{prefix}-cancel-error-message"),
                                dbc.Button(
                                    "Cancel",
                                    id=f"{prefix}-cancel-button",
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
        id=f"{prefix}-modal",
        is_open=False,
    )
