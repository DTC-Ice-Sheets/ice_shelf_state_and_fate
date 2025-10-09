"""Layout functions for the dashboard."""

import dash_bootstrap_components as dbc
from dash import MATCH, Input, Output, callback, html
from dash.development.base_component import Component


def get_navbar_pages() -> list[dbc.DropdownMenuItem]:
    """Get the list of menu items for the navigation bar."""
    return [
        dbc.DropdownMenuItem("More pages", header=True),
        dbc.DropdownMenuItem("Insights", href="/insights"),
        dbc.DropdownMenuItem("Ice shelf summary", href="/ice-shelf-summary"),
        dbc.DropdownMenuItem("Ice shelf analysis", href="/covariate-analysis"),
    ]


def layout_navbar(page_title: str) -> dbc.NavbarSimple:
    """Create a navigation bar for the dashboard.

    Parameters
    ----------
    page_title : str
        Title of the current page to display in the navbar.

    Returns
    -------
    dbc.NavbarSimple
        The navigation bar component.
    """
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(page_title, href="#")),
            dbc.DropdownMenu(
                children=get_navbar_pages(),
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand=page_title,
        brand_href="#",
        color="black",
        dark=True,
        brand_style={
            "font-size": "2rem",
            "text-align": "center",
            "font-weight": "bold",
            "margin": "auto",
        },
        sticky="sticky",
        fixed="sticky",
    )


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
