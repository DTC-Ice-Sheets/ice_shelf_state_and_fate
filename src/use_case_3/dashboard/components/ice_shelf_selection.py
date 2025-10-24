"""Ice shelf dropdown and time range selection."""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc

from use_case_3.dashboard.mapping import plot_ice_shelf_map
from use_case_3.dashboard.query_api import (
    get_ice_shelf_time_range,
    get_ice_shelves,
)


def create_ice_shelf_selection(show_map: bool = True) -> list[dbc.Row]:
    """Create the ice shelf selection dropdown and time range slider.

    Parameters
    ----------
    show_map : bool, optional
        Whether to show the ice shelf map under the dropdown, by default True.
        Intended for pages that place the map elsewhere using `create_ice_shelf_map()`.

    Returns
    -------
    list[dbc.Row]
        A list of rows intended to be placed inside a `dbc.Form`.
    """
    return [
        dbc.Row(
            [
                dbc.Label("Ice Shelf:", html_for="ice-shelf-selection", width=3),
                dbc.Col(
                    [
                        dbc.Select(
                            options=["Loading..."],
                            value="Loading...",
                            id="ice-shelf-selection",
                            style={"textTransform": "capitalize"},
                        ),
                    ]
                ),
            ]
        ),
        *([dbc.Row(create_ice_shelf_map())] if show_map else []),
        dbc.Row(
            [
                dbc.Label("Year Range:", html_for="summary-year-selector", width=3),
                dbc.Col(
                    dcc.RangeSlider(
                        0,
                        0,
                        1,
                        id="summary-year-selector",
                        marks=None,
                        tooltip={
                            "placement": "bottom",
                            "always_visible": True,
                            "style": {"font-size": "16px"},
                        },
                    ),
                    style={"padding-bottom": "8px"},
                ),
            ]
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
    Output("ice-shelf-selection", "options"),
    Output("ice-shelf-selection", "value"),
    Input("_pages_content", "id"),
)
async def load_ice_shelves(_id: str) -> tuple[list[str], str]:
    """Set up the ice shelf selection dropdown."""
    ice_shelves = await get_ice_shelves()
    initial_ice_shelf = "thwaites"
    return ice_shelves, initial_ice_shelf


@callback(
    Output("summary-year-selector", "min"),
    Output("summary-year-selector", "max"),
    Output("summary-year-selector", "value"),
    Input("ice-shelf-selection", "value"),
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
    Input("ice-shelf-selection", "value"),
)
async def update_graph(value: str) -> go.Figure:
    """Update the ice shelf map when the ice shelf selection changes."""
    return await plot_ice_shelf_map(ice_shelf=value)
