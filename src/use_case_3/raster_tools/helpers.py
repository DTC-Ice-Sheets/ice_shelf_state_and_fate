"""Gridding helper functions."""
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


def make_grid_mesh_xy(xs: NDArray[Any], ys: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    For a set of unique xs and ys coordinates, return every perumation.

    This exists as, at the time of writing, it was order of magnitudes faster than using np.meshgrid.
    TODO: Re-assert this claim.

    Parameters
    ----------
    xs : NDArray[Any]
        A unique list of x values.
    ys : NDArray[Any]
        A unique list of y values.

    Returns
    -------
    Tuple[NDArray[Any], NDArray[Any]]
        Each x in the set of x/y permutations, and each y in the set of x/y permutations.
    """
    return np.repeat(xs, len(ys)), np.tile(ys, len(xs))
