"""Layout functions for the dashboard."""

import dash_bootstrap_components as dbc
from dash import MATCH, Input, Output, callback, html
from dash.development.base_component import Component


def define_navbar_pages():
    return [
        dbc.DropdownMenuItem("More pages", header=True),
        dbc.DropdownMenuItem("Insights", href="/insights"),
        dbc.DropdownMenuItem("Ice shelf summary", href="/ice-shelf-summary"),
        dbc.DropdownMenuItem("Ice shelf analysis", href="/covariate-analysis"),
    ]


def layout_navbar(page_title):
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(page_title, href="#")),
            dbc.DropdownMenu(
                children=define_navbar_pages(),
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
):
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
def group_visibility(is_visible):
    if is_visible:
        return {"display": "block"}
    return {"display": "none"}
