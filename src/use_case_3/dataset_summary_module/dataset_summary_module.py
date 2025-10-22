"""Simple module to produce a quick plottable summary of an xarray dataset."""

import numpy as np
import xarray as xr


def calculate_summary_grid(data: xr.Dataset, max_res: int = 50) -> dict:
    """Calculate a multi-dimensional grid summary of an xarray dataset.

    Parameters
    ----------
    data : xr.Dataset
        The input xarray dataset to summarize.
    max_res : int, optional
        The maximum number of points to sample along each dimension.
        If a dimension size is less than or equal to max_res, all points are used.

    Returns
    -------
    dict
        A dictionary containing the summary grid.
        - "dims":
            A dictionary where each key is a dimension name and the value is another
            dictionary with:
                - "real_size": The actual size of the original dimension.
                - "idxs": The indices sampled along this dimension to produce the grid.
                - "values": The coordinate values at the sampled indices.
        - "vars":
            A dictionary where each key is a variable name and the value is another
            dictionary with:
                - "coords": The list of coordinate names for the variable.
                - "values": The values of the variable at the sampled indices.
                            This has dimensionality equal to the length of `coords`.
    """
    out = {
        "dims": {},
        "vars": {},
    }
    for name, size in data.dims.items():
        idxs = np.linspace(0, size - 1, min(size, max_res)).astype(int)
        values = data[name][idxs].values
        out["dims"][name] = {
            "real_size": size,
            "idxs": idxs.tolist(),
            "values": values.tolist(),
        }
    for name, var in data.data_vars.items():
        coords = list(var.coords)
        values = var.isel({coord: out["dims"][coord]["idxs"] for coord in coords}).values
        out["vars"][name] = {
            "coords": list(var.coords),
            "values": values.tolist(),
        }
    return out
