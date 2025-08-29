import os.path as path


def get_plot_base_path() -> str:
    """
    Get the base folder path.

    Returns
    -------
    str
        The path to the investigations/plots folder
    """
    return path.join(path.dirname(path.dirname(__file__)), "investigations", "plots")


def get_plot_path(base_path: str, file_name: str) -> str:
    """
    Find the path to where the plot is stored.

    Parameters
    ----------
    base_path : str
        The base folder
    file_name : str
        The plot filename

    Returns
    -------
    str
        The full path to the plot
    """
    return path.join(base_path, file_name)


def iceshelf_polygon(ice_shelf_polygons, ice_shelf_name):
    ice_shelf_polygons_epsg4326 = ice_shelf_polygons.to_crs("epsg:4326")
    ice_shelf_polygon_epsg4326 = ice_shelf_polygons_epsg4326[ice_shelf_polygons_epsg4326["Name"] == ice_shelf_name][
        "geometry"
    ].values[0]
    ice_shelf_polygon = ice_shelf_polygons[ice_shelf_polygons["Name"] == ice_shelf_name]["geometry"].values[0]
    return ice_shelf_polygon_epsg4326, ice_shelf_polygon


def ocean_polygon(ice_shelf_polygons, ice_shelf_name, distance=10000):
    ice_shelf_polygon = ice_shelf_polygons[ice_shelf_polygons["Name"] == ice_shelf_name].copy()
    ice_shelf_polygon["geometry"] = (
        ice_shelf_polygon["geometry"].values[0].buffer(distance=distance, join_style="bevel")
    )
    ice_shelf_polygon_epsg4326 = ice_shelf_polygon.to_crs("epsg:4326")

    return ice_shelf_polygon_epsg4326["geometry"].values[0], ice_shelf_polygon["geometry"].values[0]


def climate_polygon():
    pass
