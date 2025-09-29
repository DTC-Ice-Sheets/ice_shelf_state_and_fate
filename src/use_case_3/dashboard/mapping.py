import base64
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from dash import html

BUFFER_PATH = Path("/data/yak1/working/dtc_ice_sheets/dashboard_map_buffer")


def create_ice_shelf_map(
    ice_shelf_polygons: gpd.GeoDataFrame,
    ocean_polygons: gpd.GeoDataFrame | None = None,
    ice_sheet_polygons: gpd.GeoDataFrame | None = None,
    ice_shelf: str | None = "Thwaites",
) -> tuple[plt.figure, plt.axes]:
    fig, axes = plt.subplots(figsize=(15, 15))
    if ice_shelf is not None:
        ice_shelf_polygons[ice_shelf_polygons["Name"] == ice_shelf].plot(
            ax=axes,
            color="salmon",
            alpha=1,
            linewidth=1,
            edgecolor="black",
        )
        if ocean_polygons is not None:
            ocean_polygons[ocean_polygons["Name"] == ice_shelf].plot(
                ax=axes,
                color="lightskyblue",
                alpha=0.5,
                linewidth=1,
                edgecolor="black",
            )
        if ice_sheet_polygons is not None:
            ice_sheet_polygons.plot(
                ax=axes,
                color="palegreen",
                alpha=0.5,
                linewidth=1,
                edgecolor="black",
                autolim=False,
            )
        ice_shelf_polygons.plot(
            ax=axes,
            color="silver",
            alpha=0.25,
            linewidth=1,
            edgecolor="black",
            autolim=False,
        )
    else:
        ice_shelf_polygons.plot(
            ax=axes,
            color="salmon",
            alpha=1,
            linewidth=1,
            edgecolor="black",
        )
        if ice_sheet_polygons is not None:
            ice_sheet_polygons.plot(
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
    ice_shelf_map = html.Img(src=ice_shelf_map, style={"max-width": "100%", "max-height": "100%", "border": "0px"})

    return ice_shelf_map
