"""Error banner component."""

import dash_bootstrap_components as dbc


def create_error_banner(id: str) -> dbc.Alert:  # noqa: A002
    """Create an error banner component.

    By default the banner is hidden. To show an error, set `is_open` to True and
    set the `children` property to the error message.

    Parameters
    ----------
    id : str
        The ID of the error banner.

    Returns
    -------
    dbc.Alert
        The error banner component.
    """
    return dbc.Alert(
        "Error",
        id=id,
        color="danger",
        is_open=False,
        dismissable=True,
    )
