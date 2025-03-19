"""Test the helper functions."""
import numpy as np

from use_case_3.raster_tools import helpers


def test_make_grid_mesh_xy():
    """Test the make_grid_mesh_xy function."""
    xs = np.array([0, 1, 2])
    ys = np.array([3, 4])
    xs_returned, ys_returned = helpers.make_grid_mesh_xy(xs, ys)

    np.testing.assert_array_equal(np.array([0, 0, 1, 1, 2, 2]), xs_returned)
    np.testing.assert_array_equal(np.array([3, 4, 3, 4, 3, 4]), ys_returned)
