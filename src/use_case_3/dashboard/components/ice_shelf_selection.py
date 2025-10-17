"""Ice shelf dropdown and time range selection."""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from use_case_3.dashboard.mapping import plot_ice_shelf_map
from use_case_3.dashboard.query_api import (
    get_ice_shelf_time_range,
    get_ice_shelves,
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
                style={"textTransform": "capitalize"},
            ),
            style={"border": "0px black solid", "padding-top": "10px"},
        ),
        html.Div(
            dcc.RangeSlider(
                0,
                0,
                0,
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


def create_ice_shelf_map() -> dcc.Loading:
    """Create the ice shelf map.

    Returns
    -------
    dcc.Loading
        A container that will be updated to show the ice shelf map.
    """
    return dcc.Loading(
        [dcc.Graph(id="ice_shelf_map")],
        type="default",
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
    min_time, max_time = await get_ice_shelf_time_range(ice_shelf=ice_shelf)
    min_year = min_time.year
    max_year = max_time.year

    return (
        min_year,
        max_year,
        [min_year, max_year],
    )


@callback(
    Output("ice_shelf_map", "figure"),
    Input("ice_shelf_selection", "value"),
)
async def update_graph(value: str) -> go.Figure:
    """Update the ice shelf map when the ice shelf selection changes."""
    return await plot_ice_shelf_map(ice_shelf=value)
