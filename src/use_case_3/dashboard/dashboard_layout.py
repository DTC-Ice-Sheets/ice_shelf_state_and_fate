"""Layout functions for the dashboard."""

import dash_bootstrap_components as dbc

navbar_pages = {
    "/insights": "Insights",
    "/ice-shelf-summary": "Ice Shelf Summary",
    "/covariate-analysis": "Covariate Analysis",
}


def layout_navbar(current_path: str | None = None) -> dbc.NavbarSimple:
    """Create a navigation bar for the dashboard.

    Parameters
    ----------
    current_path : str | None
        The current path of the page, used to highlight the active link.
        If None, no link will be highlighted.

    Returns
    -------
    dbc.NavbarSimple
        The navigation bar component.
    """
    return dbc.NavbarSimple(
        children=[
            *[
                dbc.NavItem(
                    dbc.NavLink(
                        name,
                        href=path,
                        active=(path == current_path),
                    )
                )
                for path, name in navbar_pages.items()
            ]
        ],
        brand="DTC Ice Sheets: Use Case 3",
        brand_href="/",
        brand_style={"font-size": "2rem"},
        color="#1c1c1c",
        dark=True,
        sticky="top",
        style={
            "font-size": "1.5rem",
            "margin-bottom": "1em",
        },
    )
