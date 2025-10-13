"""Group of components that can be collapsed or expanded with a switch."""

import dash_bootstrap_components as dbc
from dash import MATCH, Input, Output, callback, html
from dash.development.base_component import Component


def create_collapsible_group(
    group_id: str,
    group_label: str,
    group_description: str,
    content: list[Component],
) -> html.Div:
    """Create a div with content that can be shown or hidden by toggling a switch.

    Parameters
    ----------
    group_id : str
        Unique identifier for the group.
    group_label : str
        Human-readable label next to the switch.
    group_description : str
        Human-readable description below the switch.
    content : list[Component]
        Content of the div that can be shown or hidden.

    Returns
    -------
    html.Div
        A div containing the switch, description, and content.
    """
    elements = [
        dbc.Switch(
            id={"type": "group-switch", "index": group_id},
            label=group_label,
            value=False,
        ),
        html.Div(
            [
                html.P(
                    group_description,
                    className="analysis-data-desc",
                )
            ]
        ),
    ]

    if len(content) > 0:
        elements.append(
            html.Div(
                content,
                className="analysis-vars-switches",
                id={"type": "group-content", "index": group_id},
            )
        )

    return html.Div(elements)


@callback(
    Output({"type": "group-content", "index": MATCH}, "style"),
    Input({"type": "group-switch", "index": MATCH}, "value"),
)
def group_visibility(is_visible: bool) -> dict[str, str]:
    """Update group visibility when group visibility switch is toggled."""
    if is_visible:
        return {"display": "block"}
    return {"display": "none"}
