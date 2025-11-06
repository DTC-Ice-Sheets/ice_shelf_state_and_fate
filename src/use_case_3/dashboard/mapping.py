"""Map plotting functions."""

import functools
from pathlib import Path

import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dtc_query_client import IceShelfExtent
from shapely import MultiPolygon

from use_case_3.dashboard.query_api import get_ice_shelf_geometry

icesheet_polygon_path = Path("src/use_case_3/dashboard/assets/icesheet_polygon.parquet")


@functools.cache
def get_antarctica_outline() -> MultiPolygon:
    """Get a high-resolution outline of Antarctica for plotting."""
    return gpd.read_parquet(icesheet_polygon_path).to_crs("EPSG:4326").iloc[0].geometry


async def plot_ice_shelf_map(ice_shelf: str) -> go.Figure:
    """Plot a map for the given ice shelf.

    Parameters
    ----------
    ice_shelf : str
        Name of the ice shelf to plot.

    Returns
    -------
    go.Figure
        A Plotly figure displaying the map.
    """
    geom_shelf = await get_ice_shelf_geometry(ice_shelf, IceShelfExtent.ICE_SHELF)
    geom_ocean = await get_ice_shelf_geometry(ice_shelf, IceShelfExtent.OCEAN)
    geom_climate = await get_ice_shelf_geometry(ice_shelf, IceShelfExtent.CLIMATE)
    antarctica_outline = get_antarctica_outline()

    gdf = gpd.GeoDataFrame(
        {"name": ["Climate", "Ocean", "Ice Shelf"]},
        geometry=[geom_climate, geom_ocean, geom_shelf],
        crs="EPSG:4326",
    )

    minx, miny, maxx, maxy = gdf.total_bounds

    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color="name",
        color_discrete_map={"Climate": "lightgreen", "Ocean": "lightskyblue", "Ice Shelf": "salmon"},
        basemap_visible=False,
        hover_name="name",
        hover_data={"index": (False, gdf.index), "name": False},
    )

    fig.add_traces(
        [
            go.Scattergeo(
                lon=[point[0] for point in antarctica_outline.geoms[i].exterior.coords],
                lat=[point[1] for point in antarctica_outline.geoms[i].exterior.coords],
                mode="lines",
                fill="toself",
                fillcolor="darkgray",
                opacity=0.5,
                line=go.Line(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
            for i in range(len(antarctica_outline.geoms))
        ]
    )

    fig.update_geos(
        projection={
            "type": "stereographic",
            "rotation": {
                "lon": (minx + maxx) / 2,
                "lat": (miny + maxy) / 2,
                "roll": 0,
            },
            "scale": 40,
        },
    )
    fig.update_layout(
        margin={"r": 0, "t": 20, "l": 0, "b": 0},
        showlegend=True,
        legend_title=None,
    )

    return fig
