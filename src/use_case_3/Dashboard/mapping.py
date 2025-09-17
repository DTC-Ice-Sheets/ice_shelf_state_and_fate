import contextily as cx
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from io import BytesIO
import base64
from dash import html


BUFFER_PATH = Path("/data/yak1/working/dtc_ice_sheets/dashboard_map_buffer")


def create_ice_shelf_map(
    ice_shelf_polygons: gpd.GeoDataFrame, ice_shelf: str = "Thwaites"
) -> tuple[plt.figure, plt.axes]:
    fig, axes = plt.subplots(figsize=(15, 15))
    ice_shelf_polygons[ice_shelf_polygons["Name"] == ice_shelf].plot(
        ax=axes,
        color="red",
        alpha=0.5,
        linewidth=1,
        edgecolor="black",
        # aspect=None,
    )
    cx.add_basemap(axes, crs="EPSG:3031")
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
    ice_shelf_map = html.Img(src=ice_shelf_map, style={"max-width": "100%", "max-height": "100%"})
    # ice_shelf_map = html.Img(
    #     src=ice_shelf_map, style={"max-width": "80%", "max-height": "100%", "margin": "0", "padding": "0"}
    # )
    # ice_shelf_map = html.Img(src=ice_shelf_map, style={"max-width": "25%", "max-height": "25%"})

    return ice_shelf_map
