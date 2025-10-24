"""S3 Directory Download Button Component."""

import logging

import dash_bootstrap_components as dbc
from dash import MATCH, Input, Output, callback, dcc, html

from use_case_3.dashboard.components.error_banner import create_error_banner
from use_case_3.dashboard.components.progress_dialog import create_progress_dialog

logger = logging.getLogger(__name__)


def create_s3_directory_download_button(
    label: str,
    index: str,
) -> html.Div:
    """Create a button to download a directory from S3.

    The button will be hidden until the `s3-download-url` store is populated with a valid S3 URL.

    Unfortunately, due to limitations in Dash, the callback to handle the actual download cannot
    be defined here as a pattern-matching callback. Instead, it must be re-defined in each page
    that uses this component. See `covariate_analysis.py` for an example.

    Parameters
    ----------
    label : str
        The label of the button.
    index : str
        Index used for all sub-components.
        The following component types will be created with this index:
        - s3-download-error-message: An error banner to show errors related to downloading the S3 directory.
        - s3-download-modal: A progress dialog modal to show the progress of the download.
        - s3-download-url: A dcc.Store to hold the S3 URL of the directory to be downloaded.
        - s3-download-button: The button to initiate the download.
        - s3-download-download: A dcc.Download component to trigger the download of the S3 directory.

    Returns
    -------
    html.Div
        Container div holding the button and related components.
    """
    return html.Div(
        [
            create_error_banner(id={"type": "s3-download-error-message", "index": index}),
            create_progress_dialog(title="Data Download", index=index, cancellable=False),
            dcc.Store(id={"type": "s3-download-url", "index": index}, data=""),
            dbc.Button(
                label,
                id={"type": "s3-download-button", "index": index},
                color="primary",
                outline=True,
                size="lg",
            ),
            dcc.Download(id={"type": "s3-download-download", "index": index}),
        ]
    )


@callback(
    Output({"type": "s3-download-button", "index": MATCH}, "style"),
    Input({"type": "s3-download-url", "index": MATCH}, "data"),
)
def s3_directory_download_button_visible(s3_url: str) -> dict[str, str]:
    """Show download button when S3 directory is ready for download."""
    if s3_url == "":
        return {"display": "none"}
    return {"display": "block"}
