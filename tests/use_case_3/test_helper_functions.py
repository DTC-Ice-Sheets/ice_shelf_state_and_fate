"""Test the helper functions."""

import numpy as np
import pytest
import xarray as xr

from use_case_3.helper_functions import find_isotherm


@pytest.fixture
def ocean_temp_fixture():
    data = np.zeros((2, 3, 2, 2))
    data[:] = np.nan
    data[0, :, 0, 0] = [0.8, 0.8, 0.8]
    data[1, :, 0, 0] = [1, 0.5, 0]
    data[0, :, 0, 1] = [2, 1, 0]
    data[1, :, 0, 1] = [3, 2, 1]
    data[0, :, 1, 0] = [1, np.nan, np.nan]
    data[1, :, 1, 0] = [1, 0.8, 0]
    data[0, :, 1, 1] = [2, 1, 0.8]
    data[1, :, 1, 1] = [0.8, 0.5, 0]

    x = [0, 1]
    y = [0, 1]
    time = [0, 1]
    depth = [5, 10, 15]

    return xr.Dataset(
        data_vars={"thetao": (["time", "depth", "y", "x"], data)},
        coords={
            "time": (["time"], time),
            "depth": (["depth"], depth),
            "y": (["y"], y),
            "x": (["x"], x),
        },
    )


def test_isotherm(ocean_temp_fixture):
    isotherm = find_isotherm(data=ocean_temp_fixture, var_name="thetao", isotherm_temp=0.8)

    assert isotherm[0, 0, 0] == 5
    assert isotherm[1, 0, 0] == 10
    assert isotherm[0, 0, 1] == 15
    assert np.isnan(isotherm[1, 0, 1])
    assert np.isnan(isotherm[1, 0, 1])
    assert isotherm[1, 1, 0] == 10
    assert isotherm[0, 1, 1] == 15
    assert isotherm[1, 1, 1] == 5
