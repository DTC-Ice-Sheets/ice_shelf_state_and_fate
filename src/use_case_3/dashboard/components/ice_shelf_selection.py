"""Ice shelf dropdown and time range selection."""

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from use_case_3.dashboard.mapping import plot_ice_shelf_map
from use_case_3.dashboard.query_api import (
    get_ice_shelves,
    query_summary_data,
)


def create_ice_shelf_selection() -> list[html.Div]:
    """Create the ice shelf selection dropdown and time range slider.

    Returns
    -------
    list[html.Div]
        A list of divs containing the dropdown and slider.
    """
    return [
        html.Div(
            dbc.Select(
                options=["Loading..."],
                value="Loading...",
                id="ice_shelf_selection",
                style={},
            ),
            style={"border": "0px black solid", "padding-top": "10px"},
        ),
        html.Div(
            dcc.RangeSlider(
                1990,
                2025,
                1,
                id="summary_year_selector",
                marks=None,
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                    "style": {"color": "White", "fontSize": "20px"},
                },
            ),
            style={"border": "0px black solid", "padding-top": "10px"},
        ),
    ]


def create_ice_shelf_map() -> html.Div:
    """Create the ice shelf map.

    Returns
    -------
    html.Div
        A container that will be updated to show the ice shelf map.
    """
    return html.Div(
        id="ice_shelf_map",
        style={"text-align": "center", "border": "0px black solid"},
    )


@callback(
    Output("ice_shelf_selection", "options"),
    Output("ice_shelf_selection", "value"),
    Input("_pages_content", "id"),
)
async def load_ice_shelves(_id: str) -> tuple[list[str], str]:
    """Set up the ice shelf selection dropdown."""
    ice_shelves = await get_ice_shelves()
    initial_ice_shelf = "thwaites"
    return ice_shelves, initial_ice_shelf


@callback(
    Output("summary_year_selector", "min"),
    Output("summary_year_selector", "max"),
    Output("summary_year_selector", "value"),
    Input("ice_shelf_selection", "value"),
)
async def update_summary_year_selector(ice_shelf: str) -> tuple[int, int, list[int]]:
    """Update the summary year selector when the ice shelf selection changes."""
    summary_data, _ = await query_summary_data(ice_shelf)
    min_year = summary_data["time"].min().item().year
    max_year = summary_data["time"].max().item().year

    return (
        min_year,
        max_year,
        [min_year, max_year],
    )


@callback(
    Output("ice_shelf_map", "children"),
    Input("ice_shelf_selection", "value"),
)
def update_graph(value: str) -> html.Img:
    """Update the ice shelf map when the ice shelf selection changes."""
    return plot_ice_shelf_map(ice_shelf=value)
