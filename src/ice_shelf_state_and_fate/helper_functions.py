"""Helper functions for Use Case 3."""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


def get_plot_base_path() -> Path:
    """
    Get the base folder path.

    Returns
    -------
    Path
        The path to the investigations/plots folder
    """
    return Path.cwd() / "investigations" / "Plots"


def get_plot_path(base_path: Path, file_name: str) -> Path:
    """
    Find the path to where the plot is stored.

    Parameters
    ----------
    base_path : Path
        The base folder
    file_name : str
        The plot filename

    Returns
    -------
    Path
        The full path to the plot
    """
    return base_path / file_name


def iceshelf_polygon(
    ice_shelf_polygons: gpd.GeoDataFrame, ice_shelf_name: str
) -> tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """
    Extract the ice shelf multipolygon for an ice shelf in both EPSG4326(lat/lon) and the input projection.

    Parameters
    ----------
    ice_shelf_polygons : gpd.GeoDataFrame
        Input GeoDataFrame with atleas the column 'Name' for the ice shelf name and 'geometry' with the polygon. This
        GeoDataFrame must have a valid CRS.
    ice_shelf_name : str
        The ice shelf name

    Returns
    -------
    tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]
        The shapely object for the specified ice shelf in both EPSG4326(lat/lon) and the input projection.
    """
    ice_shelf_polygons_epsg4326 = ice_shelf_polygons.to_crs("epsg:4326")
    ice_shelf_polygon_epsg4326 = ice_shelf_polygons_epsg4326[ice_shelf_polygons_epsg4326["Name"] == ice_shelf_name][
        "geometry"
    ].values[0]
    ice_shelf_polygon = ice_shelf_polygons[ice_shelf_polygons["Name"] == ice_shelf_name]["geometry"].values[0]
    return ice_shelf_polygon_epsg4326, ice_shelf_polygon


def ocean_polygon(
    ice_shelf_polygons: gpd.GeoDataFrame, ice_shelf_name: str, distance: int | float = 10000
) -> tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """
    Extract the ice shelf ocean multipolygon for an ice shelf in both EPSG4326(lat/lon) and the input projection. To do
    this a buffer of 'distance' units is applied to the ice shelf polygon.

    Parameters
    ----------
    ice_shelf_polygons : gpd.GeoDataFrame
        Input GeoDataFrame with atleas the column 'Name' for the ice shelf name and 'geometry' with the polygon. This
        GeoDataFrame must have a valid CRS.
    ice_shelf_name : str
        The ice shelf name
    distance : int | float
        The distance in projected units to buffer the ice shelf polygon, by default 10000

    Returns
    -------
    tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]
        The shapely object for the ocean around the specified ice shelf in both EPSG4326(lat/lon) and the input
        projection.
    """  # noqa: D205
    ice_shelf_polygon = ice_shelf_polygons[ice_shelf_polygons["Name"] == ice_shelf_name].copy()
    ice_shelf_polygon["geometry"] = (
        ice_shelf_polygon["geometry"].values[0].buffer(distance=distance, join_style="bevel")
    )
    ice_shelf_polygon_epsg4326 = ice_shelf_polygon.to_crs("epsg:4326")

    return ice_shelf_polygon_epsg4326["geometry"].values[0], ice_shelf_polygon["geometry"].values[0]
