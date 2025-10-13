"""Layout functions for the dashboard."""

import dash_bootstrap_components as dbc


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
