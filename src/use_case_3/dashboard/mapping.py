"""Map plotting functions."""

import base64
import functools
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from dash import html

from use_case_3.dashboard.query_api import query_ice_sheet_data, query_polygons

BUFFER_PATH = Path("/data/yak1/working/dtc_ice_sheets/dashboard_map_buffer")


@functools.lru_cache(8)
def plot_ice_shelf_map(ice_shelf: str | None = "Thwaites") -> html.Img:
    """Plot a map for the given ice shelf.

    Parameters
    ----------
    ice_shelf : str | None
        Name of the ice shelf to plot.

    Returns
    -------
    html.Img
        A Dash HTML image component containing the map.
    """
    ice_sheet_polygons = query_ice_sheet_data()
    ice_shelf_polygons, ocean_polygons = query_polygons()

    fig, axes = plt.subplots(figsize=(15, 15))
    if ice_shelf is not None:
        ice_shelf_polygons.proj[ice_shelf_polygons.proj["Name"] == ice_shelf].plot(
            ax=axes,
            color="salmon",
            alpha=1,
            linewidth=1,
            edgecolor="black",
        )
        if ocean_polygons is not None:
            ocean_polygons.proj[ocean_polygons.proj["Name"] == ice_shelf].plot(
                ax=axes,
                color="lightskyblue",
                alpha=0.5,
                linewidth=1,
                edgecolor="black",
            )
        if ice_sheet_polygons is not None:
            ice_sheet_polygons.proj.plot(
                ax=axes,
                color="palegreen",
                alpha=0.5,
                linewidth=1,
                edgecolor="black",
                autolim=False,
            )
        ice_shelf_polygons.proj.plot(
            ax=axes,
            color="silver",
            alpha=0.25,
            linewidth=1,
            edgecolor="black",
            autolim=False,
        )
    else:
        ice_shelf_polygons.proj.plot(
            ax=axes,
            color="lightblue",
            alpha=1,
            linewidth=1,
            edgecolor="black",
        )
        if ice_sheet_polygons is not None:
            ice_sheet_polygons.proj.plot(
                ax=axes,
                color="lightgrey",
                alpha=0.5,
                linewidth=1,
                edgecolor="black",
                autolim=False,
            )
        axes.axes.set_axis_off()

    # cx.add_basemap(axes, crs="EPSG:3031")
    plt.grid()
    plt.xlabel("")
    plt.ylabel("")
    axes.axes.get_xaxis().set_ticks([])
    axes.axes.get_yaxis().set_ticks([])

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image_string = base64.b64encode(buffer.getvalue()).decode()

    ice_shelf_map = "data:image/png;base64," + image_string
    return html.Img(src=ice_shelf_map, style={"max-width": "100%", "max-height": "100%", "border": "0px"})
