"""Module for testing TimeSeriesModule."""

import numpy as np
import pytest

from use_case_3.timeseries_module.timeseries_module import TimeSeriesModule


@pytest.fixture
def time_series_module_fixture():
    return TimeSeriesModule()


@pytest.fixture
def data_array_fixture():
    nested_list = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
    ]
    return np.array(nested_list)


@pytest.fixture
def data_array_nans_fixture():
    nested_list = [
        [[np.nan, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, np.nan, 12], [13, 14, 15], [16, np.nan, 18]],
        [[19, 20, np.nan], [np.nan, 23, 24], [np.nan, 26, 27]],
    ]
    return np.array(nested_list)


@pytest.fixture
def dimension_name_fixture():
    return ["X", "Y", "time"]


@pytest.fixture
def dimension_data_fixture():
    return {"X": np.array([0, 1, 2]), "Y": np.array([0, 1, 2]), "time": np.array([10, 11, 12])}


def test_move_axis_time(data_array_fixture, dimension_name_fixture):
    data_reordered, dimensions_reordered = TimeSeriesModule._move_axes_time(
        data=data_array_fixture, dimension_names=dimension_name_fixture
    )

    assert dimensions_reordered == ["time", "X", "Y"]
    assert data_array_fixture[0, 0, 0] == data_reordered[0, 0, 0]
    assert data_array_fixture[0, 0, 2] == data_reordered[2, 0, 0]
    assert data_array_fixture[2, 1, 0] == data_reordered[0, 2, 1]


def test_count_nan_elements(data_array_fixture, data_array_nans_fixture):
    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_fixture, dimension_count=0, non_nan=False),
        np.array([0, 0, 0]),
    )

    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_fixture, dimension_count=0, non_nan=True),
        np.array([9, 9, 9]),
    )

    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_nans_fixture, dimension_count=0, non_nan=False),
        np.array([1, 2, 3]),
    )

    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_nans_fixture, dimension_count=0, non_nan=True),
        np.array([8, 7, 6]),
    )


def test_aggregate_statistic(time_series_module_fixture, data_array_nans_fixture, dimension_name_fixture):
    data_reordered, dimensions_reordered = time_series_module_fixture._move_axes_time(
        data=data_array_nans_fixture, dimension_names=dimension_name_fixture
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(data_reordered, "mean", dimension_names=dimensions_reordered),
        np.array([11.5, 14, 14.25]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(data_reordered, "median", dimension_names=dimensions_reordered),
        np.array([11.5, 14, 13.5]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(data_reordered, "max", dimension_names=dimensions_reordered),
        np.array([19, 26, 27]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(data_reordered, "min", dimension_names=dimensions_reordered),
        np.array([4, 2, 3]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(data_reordered, "std", dimension_names=dimensions_reordered),
        np.array([5.123, 8.635, 7.902]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(data_reordered, "IQR", dimension_names=dimensions_reordered),
        np.array([7.5, 15, 11.25]),
        rtol=1e-2,
    )


def test_aggregate_uncertainties_mean(time_series_module_fixture, data_array_nans_fixture, dimension_name_fixture):
    data_reordered, dimensions_reordered = time_series_module_fixture._move_axes_time(
        data=data_array_nans_fixture, dimension_names=dimension_name_fixture
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_uncertainties_mean(
            data_reordered, "std", dimension_names=dimensions_reordered
        ),
        np.array([5.140, 6.217, 5.761]),
        rtol=1e-2,
    )


def test_calculate_aggregate_variable_3d_mean(
    time_series_module_fixture, data_array_nans_fixture, dimension_name_fixture, dimension_data_fixture
):
    data_df = time_series_module_fixture.calculate_aggregate_variable(
        data=data_array_nans_fixture,
        dimension_data=dimension_data_fixture,
        dimension_names=dimension_name_fixture,
        aggregate="mean",
    )

    assert data_df.index.to_list() == [10, 11, 12]
    assert list(data_df.columns) == ["mean", "std", "n"]
    assert np.allclose(data_df["mean"].values, np.array([11.5, 14, 14.25]), rtol=1e-2)
    assert np.allclose(data_df["std"].values, np.array([5.123, 8.635, 7.902]), rtol=1e-2)
    assert np.allclose(data_df["n"].values, np.array([6, 7, 8]), rtol=1e-2)


def test_calculate_aggregate_variable_3d_median(
    time_series_module_fixture, data_array_nans_fixture, dimension_name_fixture, dimension_data_fixture
):
    data_df = time_series_module_fixture.calculate_aggregate_variable(
        data=data_array_nans_fixture,
        dimension_data=dimension_data_fixture,
        dimension_names=dimension_name_fixture,
        aggregate="median",
    )

    assert data_df.index.to_list() == [10, 11, 12]
    assert list(data_df.columns) == ["median", "IQR", "n"]


def test_propagate_uncertainties():
    pass
